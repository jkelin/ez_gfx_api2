use super::{
    AllocationRequest, BufferTransfer, CompletionToken, ContextIdentity, EzGfxResult,
    GeometryError, HalError, HashMap, LifecycleError, MemoryAllocator, NativeAllocation,
    NativeContext, NativeTexture, PackedHandle,
};

pub(super) fn map_frame(error: &ez_gfx_runtime::frame::FrameError) -> EzGfxResult {
    match error {
        ez_gfx_runtime::frame::FrameError::NotRecording
        | ez_gfx_runtime::frame::FrameError::MissingGraph
        | ez_gfx_runtime::frame::FrameError::NotSubmitted => EzGfxResult::NotReady,
        _ => EzGfxResult::InvalidArgument,
    }
}

pub(super) fn map_texture(error: ez_gfx_runtime::texture::TextureError) -> EzGfxResult {
    use ez_gfx_runtime::texture::TextureError;
    match error {
        TextureError::Unsupported => EzGfxResult::Unsupported,
        TextureError::NotReady => EzGfxResult::NotReady,
        TextureError::TooLarge
        | TextureError::CapacityExceeded
        | TextureError::GenerationExhausted => EzGfxResult::NativeFailure,
        _ => EzGfxResult::InvalidArgument,
    }
}

pub(super) fn destroy_native_texture(
    context: &mut NativeContext,
    texture: NativeTexture,
) -> Result<(), ez_gfx_hal::AllocationError> {
    match (context, texture) {
        (NativeContext::Vulkan(context), NativeTexture::Vulkan(texture)) => {
            context.destroy_texture(texture)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeTexture::Dx12(texture)) => {
            context.destroy_texture(texture)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeTexture::Metal(texture)) => {
            context.destroy_texture(texture)
        }
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}

pub(super) fn native_layouts(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
) -> Result<Vec<ez_gfx_hal::ShaderBufferLayout>, HalError> {
    layout
        .requirements()
        .iter()
        .map(|requirement| {
            ez_gfx_hal::ShaderBufferLayout::new(
                requirement.space,
                requirement.binding,
                requirement.descriptor_count,
                requirement.writable,
            )
        })
        .collect()
}

pub(super) fn pipeline_layout_key(
    layouts: &[ez_gfx_hal::ShaderBufferLayout],
) -> Vec<ez_gfx_hal::ShaderBufferLayout> {
    // Reflection order is not semantic; physical descriptor coordinates define interface identity.
    let mut key = layouts.to_vec();
    key.sort_unstable_by_key(|layout| {
        (
            layout.space,
            layout.binding,
            layout.descriptor_count,
            layout.writable,
        )
    });
    key
}

pub(super) fn vulkan_bindings<'a>(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    allocations: &'a HashMap<PackedHandle, (u64, NativeAllocation)>,
) -> Result<Vec<ez_gfx_backend_vulkan::NativeBufferBinding<'a>>, HalError> {
    let mut native = Vec::new();
    for requirement in layout.requirements() {
        let public = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(HalError::InvalidArgument)?;
        if requirement.descriptor_count != 1 {
            return Err(HalError::Unsupported);
        }
        let handle = match public.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Structured(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(_) => {
                return Err(HalError::Unsupported);
            }
        };
        let (size, allocation) = allocations.get(&handle).ok_or(HalError::InvalidArgument)?;
        #[cfg(not(any(windows, target_vendor = "apple")))]
        let NativeAllocation::Vulkan(allocation) = allocation;
        #[cfg(any(windows, target_vendor = "apple"))]
        let NativeAllocation::Vulkan(allocation) = allocation else {
            return Err(HalError::InvalidArgument);
        };
        native.push(ez_gfx_backend_vulkan::NativeBufferBinding {
            allocation,
            offset: 0,
            range: *size,
            writable: requirement.writable,
        });
    }
    Ok(native)
}

#[cfg(target_vendor = "apple")]
pub(super) fn metal_bindings<'a>(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    allocations: &'a HashMap<PackedHandle, (u64, NativeAllocation)>,
) -> Result<Vec<ez_gfx_backend_metal::native::NativeBufferBinding<'a>>, HalError> {
    let mut native = Vec::new();
    for requirement in layout.requirements() {
        let public = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(HalError::InvalidArgument)?;
        if requirement.descriptor_count != 1 {
            return Err(HalError::Unsupported);
        }
        let handle = match public.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Structured(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(_) => {
                return Err(HalError::Unsupported);
            }
        };
        let (_, allocation) = allocations.get(&handle).ok_or(HalError::InvalidArgument)?;
        let NativeAllocation::Metal(allocation) = allocation else {
            return Err(HalError::InvalidArgument);
        };
        native.push(ez_gfx_backend_metal::native::NativeBufferBinding {
            allocation,
            offset: 0,
            index: requirement.binding as usize,
        });
    }
    Ok(native)
}

#[cfg(windows)]
pub(super) fn dx12_bindings<'a>(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    allocations: &'a HashMap<PackedHandle, (u64, NativeAllocation)>,
) -> Result<Vec<ez_gfx_backend_dx12::native::NativeBufferBinding<'a>>, HalError> {
    let mut native = Vec::new();
    for requirement in layout.requirements() {
        let public = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(HalError::InvalidArgument)?;
        if requirement.descriptor_count != 1 {
            return Err(HalError::Unsupported);
        }
        let handle = match public.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Structured(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(_) => {
                return Err(HalError::Unsupported);
            }
        };
        let (_, allocation) = allocations.get(&handle).ok_or(HalError::InvalidArgument)?;
        let NativeAllocation::Dx12(allocation) = allocation else {
            return Err(HalError::InvalidArgument);
        };
        native.push(ez_gfx_backend_dx12::native::NativeBufferBinding {
            allocation,
            offset: 0,
            writable: requirement.writable,
        });
    }
    Ok(native)
}

pub(super) fn wait_native_idle(context: &mut NativeContext) -> Result<(), HalError> {
    match context {
        NativeContext::Vulkan(context) => context.wait_idle(),
        #[cfg(windows)]
        NativeContext::Dx12(context) => context.wait_idle(),
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.wait_idle(),
    }
}

pub(super) fn allocate_native(
    context: &mut NativeContext,
    request: AllocationRequest,
) -> Result<NativeAllocation, ez_gfx_hal::AllocationError> {
    match context {
        NativeContext::Vulkan(context) => context.allocate(request).map(NativeAllocation::Vulkan),
        #[cfg(windows)]
        NativeContext::Dx12(context) => context.allocate(request).map(NativeAllocation::Dx12),
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.allocate(request).map(NativeAllocation::Metal),
    }
}

pub(super) fn write_native(
    context: &mut NativeContext,
    allocation: &mut NativeAllocation,
    bytes: &[u8],
) -> Result<(), ez_gfx_hal::AllocationError> {
    match (context, allocation) {
        (NativeContext::Vulkan(context), NativeAllocation::Vulkan(allocation)) => {
            let target = context.mapped_slice_mut(allocation)?;
            if bytes.len() > target.len() {
                return Err(ez_gfx_hal::AllocationError::NativeFailure);
            }
            target[..bytes.len()].copy_from_slice(bytes);
            context.flush(allocation, 0, bytes.len() as u64)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeAllocation::Dx12(allocation)) => {
            let target = context.mapped_slice_mut(allocation)?;
            if bytes.len() > target.len() {
                return Err(ez_gfx_hal::AllocationError::NativeFailure);
            }
            target[..bytes.len()].copy_from_slice(bytes);
            context.flush(allocation, 0, bytes.len() as u64)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeAllocation::Metal(allocation)) => {
            let target = context.mapped_slice_mut(allocation)?;
            if bytes.len() > target.len() {
                return Err(ez_gfx_hal::AllocationError::NativeFailure);
            }
            target[..bytes.len()].copy_from_slice(bytes);
            context.flush(allocation, 0, bytes.len() as u64)
        }
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}

pub(super) fn copy_native(
    context: &mut NativeContext,
    source: &NativeAllocation,
    destination: &NativeAllocation,
    source_offset: u64,
    destination_offset: u64,
    size: u64,
) -> Result<CompletionToken, ez_gfx_hal::AllocationError> {
    match (context, source, destination) {
        (
            NativeContext::Vulkan(context),
            NativeAllocation::Vulkan(source),
            NativeAllocation::Vulkan(destination),
        ) => context.copy_buffer(source, destination, source_offset, destination_offset, size),
        #[cfg(windows)]
        (
            NativeContext::Dx12(context),
            NativeAllocation::Dx12(source),
            NativeAllocation::Dx12(destination),
        ) => context.copy_buffer(source, destination, source_offset, destination_offset, size),
        #[cfg(target_vendor = "apple")]
        (
            NativeContext::Metal(context),
            NativeAllocation::Metal(source),
            NativeAllocation::Metal(destination),
        ) => context.copy_buffer(source, destination, source_offset, destination_offset, size),
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}

pub(super) fn completed_transfer_native(
    context: &mut NativeContext,
) -> Result<u64, ez_gfx_hal::AllocationError> {
    let completed = match context {
        NativeContext::Vulkan(context) => context.completed_transfer_value()?,
        #[cfg(windows)]
        NativeContext::Dx12(context) => context.completed_transfer_value()?,
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.completed_transfer_value()?,
    };
    match context {
        NativeContext::Vulkan(context) => {
            context.reclaim(ez_gfx_hal::QueueKind::Transfer, completed)?;
        }
        #[cfg(windows)]
        NativeContext::Dx12(context) => {
            context.reclaim(ez_gfx_hal::QueueKind::Transfer, completed)?;
        }
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => {
            context.reclaim(ez_gfx_hal::QueueKind::Transfer, completed)?;
        }
    }
    Ok(completed)
}

pub(super) fn completed_texture_transfer_native(
    context: &mut NativeContext,
) -> Result<u64, ez_gfx_hal::AllocationError> {
    let completed = match context {
        NativeContext::Vulkan(context) => context.completed_texture_transfer_value()?,
        #[cfg(windows)]
        NativeContext::Dx12(context) => context.completed_texture_transfer_value()?,
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.completed_texture_transfer_value()?,
    };
    match context {
        NativeContext::Vulkan(context) => {
            context.reclaim(ez_gfx_hal::QueueKind::TextureTransfer, completed)?;
        }
        #[cfg(windows)]
        NativeContext::Dx12(context) => {
            context.reclaim(ez_gfx_hal::QueueKind::TextureTransfer, completed)?;
        }
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => {
            context.reclaim(ez_gfx_hal::QueueKind::TextureTransfer, completed)?;
        }
    }
    Ok(completed)
}

pub(super) fn free_native_allocation(
    context: &mut NativeContext,
    allocation: NativeAllocation,
) -> Result<(), ez_gfx_hal::AllocationError> {
    match (context, allocation) {
        (NativeContext::Vulkan(context), NativeAllocation::Vulkan(allocation)) => {
            context.free(allocation)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeAllocation::Dx12(allocation)) => {
            context.free(allocation)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeAllocation::Metal(allocation)) => {
            context.free(allocation)
        }
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}
pub(super) fn map_allocation(error: ez_gfx_hal::AllocationError) -> EzGfxResult {
    match error {
        ez_gfx_hal::AllocationError::ZeroSize
        | ez_gfx_hal::AllocationError::InvalidAlignment
        | ez_gfx_hal::AllocationError::NotHostVisible
        | ez_gfx_hal::AllocationError::InvalidAliasClass => EzGfxResult::InvalidArgument,
        ez_gfx_hal::AllocationError::DeviceLost => EzGfxResult::DeviceLost,
        ez_gfx_hal::AllocationError::Unsupported => EzGfxResult::Unsupported,
        ez_gfx_hal::AllocationError::OutOfMemory | ez_gfx_hal::AllocationError::NativeFailure => {
            EzGfxResult::NativeFailure
        }
    }
}

pub(super) fn map_geometry(error: GeometryError) -> EzGfxResult {
    match error {
        GeometryError::StagingPoolExhausted => EzGfxResult::NotReady,
        _ => EzGfxResult::InvalidArgument,
    }
}

pub(super) fn map_lifecycle(error: LifecycleError) -> EzGfxResult {
    match error {
        LifecycleError::DeviceLost | LifecycleError::AlreadyLost => EzGfxResult::DeviceLost,
        _ => EzGfxResult::InvalidContext,
    }
}
pub(super) fn map_native_loss(identity: &ContextIdentity, error: HalError) -> EzGfxResult {
    if error == HalError::DeviceLost {
        let _ = identity.mark_lost();
    }
    map_hal(error)
}
pub(super) fn result_status(result: Result<(), EzGfxResult>) -> EzGfxResult {
    match result {
        Ok(()) => EzGfxResult::Ok,
        Err(status) => status,
    }
}
pub(super) fn map_hal(error: HalError) -> EzGfxResult {
    match error {
        HalError::InvalidArgument => EzGfxResult::InvalidArgument,
        HalError::Unsupported => EzGfxResult::Unsupported,
        HalError::NotReady => EzGfxResult::NotReady,
        HalError::DeviceLost => EzGfxResult::DeviceLost,
        HalError::OutOfMemory | HalError::NativeFailure => EzGfxResult::NativeFailure,
    }
}
