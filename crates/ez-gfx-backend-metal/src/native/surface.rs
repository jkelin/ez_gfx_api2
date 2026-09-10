use super::{
    CAMetalDrawable, DeferredResource, HalError, MTLCommandBuffer, MTLCommandQueue,
    MTLCompareFunction, MTLDepthStencilDescriptor, MTLDevice, MTLPixelFormat, MTLStorageMode,
    MTLTextureDescriptor, MTLTextureUsage, NativeContext, NativeSurface, PresentationMode,
    ProtocolObject, SurfaceDepth, ThreadBound, map_allocation_hal,
};

impl NativeContext {
    pub(super) fn ensure_surface_depth(
        &mut self,
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
        // SAFETY: `texture2DDescriptorWithPixelFormat_width_height_mipmapped` receives the defined `Depth32Float` format, `u32` dimensions losslessly widened to `usize`, and no pointer arguments; its returned descriptor is retained in `descriptor`.
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
        let depth = SurfaceDepth {
            texture: ThreadBound::new(texture),
            state: ThreadBound::new(state),
            extent,
        };
        if let Some(stale) = surface.depth.replace(depth) {
            self.defer_resource(DeferredResource::Depth(stale))
                .map_err(map_allocation_hal)?;
        }
        Ok(())
    }

    /// Returns the admitted Metal device.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Acquires and presents one drawable from the borrowed `CAMetalLayer`; zero extent is minimized.
    ///
    /// # Errors
    ///
    /// Returns an error when the surface is minimized or Metal cannot acquire or submit a drawable.
    pub fn acquire_present(
        &mut self,
        surface: &mut NativeSurface,
        width: u32,
        height: u32,
        presentation_mode: PresentationMode,
    ) -> Result<(), HalError> {
        if width == 0 || height == 0 {
            return Err(HalError::NotReady);
        }
        surface.set_presentation_mode(presentation_mode)?;
        let (slot, must_wait) = self.frame_tracker.acquire();
        if must_wait {
            self.complete_frame_slot(slot)?;
        }
        let layer = surface.metal_layer();
        layer.setDevice(Some(&self.device));
        let drawable = layer.nextDrawable().ok_or(HalError::NotReady)?;
        let command = self.queue.commandBuffer().ok_or(HalError::NativeFailure)?;
        let drawable = <ProtocolObject<dyn CAMetalDrawable> as AsRef<
            ProtocolObject<dyn objc2_metal::MTLDrawable>,
        >>::as_ref(&*drawable);
        command.presentDrawable(drawable);
        self.drain_complete = false;
        command.commit();

        self.frame_slots[slot].command = Some(ThreadBound::new(command));
        self.frame_tracker.mark_submitted(slot);
        Ok(())
    }

    /// Retires backend-owned state after severing every borrowed host relationship.
    ///
    /// P-017 differs from Vulkan and DX12 here: deferred retirement retains only an independently
    /// owned `CAMetalLayer` and GPU resources, never the `NSView`. A backend-created observer layer
    /// weakly references the root layer, and removal severs that relationship synchronously.
    /// Therefore returning `true` permits immediate host release even while layer retirement waits
    /// for submitted work; later context abandonment leaks only independently owned objects.
    pub fn destroy_surface(&mut self, mut surface: NativeSurface) -> bool {
        // Host-provided CAMetalLayers are independently retained. Backend-created observer layers
        // weakly reference the root layer; removal severs their only host relationship.
        surface.detach_from_host();
        // Surface retirement performs no allocator operation and is therefore infallible.
        let _ = self.defer_resource(DeferredResource::Surface(surface));
        true
    }
}
