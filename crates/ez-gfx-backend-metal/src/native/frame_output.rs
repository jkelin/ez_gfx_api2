//! Metal frame drawable acquisition, readback allocation, and completion.

use super::frame::{MetalDrawable, MetalFrameReadback};
use super::{
    AllocationRequest, CAMetalDrawable, HalError, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLPixelFormat, MTLTexture, MemoryAllocator, MemoryClass, NativeAllocation, NativeContext,
    NativeSurface, ProtocolObject, ThreadBound, map_allocation_hal,
};

impl NativeContext {
    /// Level-zero extent of a published Metal texture view.
    ///
    /// The view covers the `resident_mips` coarse tail, so its level zero is storage
    /// mip `mip_count - resident_mips`. Returns `None` when no level is published.
    pub(super) fn published_view_extent(
        width: u32,
        height: u32,
        mip_count: u32,
        resident_mips: u32,
    ) -> Option<(usize, usize)> {
        if resident_mips == 0 || resident_mips > mip_count {
            return None;
        }
        let level = mip_count - resident_mips;
        let extent = |base: u32| {
            u64::from(base)
                .checked_shr(level)
                .map(|value| value.max(1))
                .and_then(|value| usize::try_from(value).ok())
        };
        Some((extent(width)?, extent(height)?))
    }
    pub(super) fn allocate_frame_readback(
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

    pub(super) fn finish_frame(
        &mut self,
        command: super::Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        readbacks: Vec<MetalFrameReadback>,
        slot_index: usize,
        capture_presented: bool,
        mut surface: Option<&mut NativeSurface>,
    ) -> Result<Vec<Vec<u8>>, HalError> {
        let frame_value = self.next_frame_value;
        self.next_frame_value = frame_value.checked_add(1).ok_or(HalError::NativeFailure)?;
        self.drain_complete = false;
        command.commit();
        self.last_frame_value = frame_value;
        if readbacks.is_empty() {
            self.frame_slots[slot_index].command = Some(ThreadBound::new(command));
            self.frame_slots[slot_index].submission_value = frame_value;
            self.frame_tracker.mark_submitted(slot_index);
            return Ok(Vec::new());
        }
        command.waitUntilCompleted();
        self.completed_frame_value = self.completed_frame_value.max(frame_value);
        if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
            for (allocation, _, _, _, _) in readbacks {
                let _ = self.free(allocation);
            }
            return Err(HalError::NativeFailure);
        }
        let mut outputs = Vec::with_capacity(readbacks.len());
        let mut remaining = readbacks.into_iter();
        while let Some((mut allocation, tight_row, row_stride, size, height)) = remaining.next() {
            let copied = (|| {
                self.invalidate(&mut allocation, 0, size)
                    .map_err(map_allocation_hal)?;
                let source = self.mapped_slice(&allocation).map_err(map_allocation_hal)?;
                let packed_size = tight_row
                    .checked_mul(u64::from(height))
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or(HalError::InvalidArgument)?;
                let mut packed = Vec::with_capacity(packed_size);
                for row in 0..usize::try_from(height).map_err(|_| HalError::InvalidArgument)? {
                    let start = row
                        .checked_mul(
                            usize::try_from(row_stride).map_err(|_| HalError::InvalidArgument)?,
                        )
                        .ok_or(HalError::InvalidArgument)?;
                    let end = start
                        .checked_add(
                            usize::try_from(tight_row).map_err(|_| HalError::InvalidArgument)?,
                        )
                        .ok_or(HalError::InvalidArgument)?;
                    packed
                        .extend_from_slice(source.get(start..end).ok_or(HalError::NativeFailure)?);
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
                    .presented_rgba8
                    .clone_from(&packed);
            }
            outputs.push(packed);
        }
        Ok(outputs)
    }

    pub(super) fn prepare_drawable(
        &self,
        surface: Option<&NativeSurface>,
        extent: (u32, u32),
        uses_surface: bool,
        capture_presented: bool,
    ) -> Result<Option<MetalDrawable>, HalError> {
        if uses_surface {
            let surface = surface.ok_or(HalError::InvalidArgument)?;
            let layer = surface.metal_layer();
            layer.setDevice(Some(&self.device));
            layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
            // Core Animation defaults to framebuffer-only drawables. Disable that restriction
            // before the first requested capture; leaving it disabled supports later one-frame
            // captures without recreating the host layer.
            if capture_presented {
                layer.setFramebufferOnly(false);
            }
            let drawable = layer.nextDrawable().ok_or(HalError::NotReady)?;
            let texture = drawable.texture();
            if texture.width()
                != usize::try_from(extent.0).map_err(|_| HalError::InvalidArgument)?
                || texture.height()
                    != usize::try_from(extent.1).map_err(|_| HalError::InvalidArgument)?
            {
                return Err(HalError::NotReady);
            }
            Ok(Some(drawable))
        } else {
            Ok(None)
        }
    }
}
