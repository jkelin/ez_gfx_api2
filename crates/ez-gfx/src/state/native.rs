use super::{
    AllocationRequest, BufferTransfer, COUNTER_BUFFER_ELEMENT_OFFSET, CompletionToken,
    ContextIdentity, Error, GeometryAllocation, GeometryError, HalError, HashMap, LifecycleError,
    MemoryAllocator, NativeAllocation, NativeContext, NativeTexture, PackedHandle,
};

pub(super) fn map_frame(error: &ez_gfx_runtime::frame::FrameError) -> Error {
    match error {
        ez_gfx_runtime::frame::FrameError::AlreadyRecording
        | ez_gfx_runtime::frame::FrameError::NotRecording
        | ez_gfx_runtime::frame::FrameError::MissingGraph
        | ez_gfx_runtime::frame::FrameError::NotSubmitted => Error::NotReady,
        _ => Error::InvalidArgument,
    }
}

pub(super) fn map_texture(error: ez_gfx_runtime::texture::TextureError) -> Error {
    use ez_gfx_runtime::texture::TextureError;
    match error {
        TextureError::Unsupported => Error::Unsupported,
        TextureError::NotReady => Error::NotReady,
        TextureError::TooLarge
        | TextureError::CapacityExceeded
        | TextureError::GenerationExhausted => Error::NativeFailure,
        _ => Error::InvalidArgument,
    }
}

pub(super) fn destroy_native_texture(
    context: &mut NativeContext,
    texture: NativeTexture,
) -> std::result::Result<(), ez_gfx_hal::AllocationError> {
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

/// Reports adapter compression support for capability-gated target selection.
pub(super) fn native_texture_compression(
    context: &NativeContext,
) -> ez_gfx_core::capability::CompressionSupport {
    use ez_gfx_core::capability::CompressionSupport;
    match context {
        NativeContext::Vulkan(native) => native
            .adapter_info()
            .map_or(CompressionSupport::NONE, |adapter| {
                adapter.capabilities().compression
            }),
        #[cfg(windows)]
        NativeContext::Dx12(native) => native.adapter_info().capabilities().compression,
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(native) => native.adapter_info().capabilities().compression,
    }
}

pub(super) fn native_layouts(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
) -> std::result::Result<Vec<ez_gfx_hal::ShaderBufferLayout>, HalError> {
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
    vertex_heaps: &'a HashMap<String, GeometryAllocation>,
) -> std::result::Result<Vec<ez_gfx_backend_vulkan::NativeBufferBinding<'a>>, HalError> {
    let mut native = Vec::new();
    for requirement in layout.requirements() {
        if requirement.kind == ez_gfx_runtime::binding::BindingKind::VertexHeap {
            if requirement.descriptor_count != 1 {
                return Err(HalError::Unsupported);
            }
            let heap = vertex_heaps
                .get(&requirement.name)
                .ok_or(HalError::InvalidArgument)?;
            #[cfg(not(any(windows, target_vendor = "apple")))]
            let NativeAllocation::Vulkan(allocation) = &heap.allocation;
            #[cfg(any(windows, target_vendor = "apple"))]
            let NativeAllocation::Vulkan(allocation) = &heap.allocation else {
                return Err(HalError::InvalidArgument);
            };
            native.push(ez_gfx_backend_vulkan::NativeBufferBinding {
                allocation,
                offset: 0,
                range: heap.size,
                writable: requirement.writable,
            });
            continue;
        }
        let public = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(HalError::InvalidArgument)?;
        let handle = match public.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Buffer(handle)
                if requirement.kind == ez_gfx_runtime::binding::BindingKind::Buffer
                    && requirement.descriptor_count == 1 =>
            {
                handle.packed()
            }
            ez_gfx_runtime::binding::ResourceIdentity::Counter(handle)
                if requirement.kind == ez_gfx_runtime::binding::BindingKind::CounterBuffer
                    && requirement.descriptor_count == 2 =>
            {
                handle.packed()
            }
            _ => return Err(HalError::Unsupported),
        };
        let (size, allocation) = allocations.get(&handle).ok_or(HalError::InvalidArgument)?;
        #[cfg(not(any(windows, target_vendor = "apple")))]
        let NativeAllocation::Vulkan(allocation) = allocation;
        #[cfg(any(windows, target_vendor = "apple"))]
        let NativeAllocation::Vulkan(allocation) = allocation else {
            return Err(HalError::InvalidArgument);
        };
        if requirement.descriptor_count == 2 {
            let command_size = size
                .checked_sub(COUNTER_BUFFER_ELEMENT_OFFSET)
                .ok_or(HalError::InvalidArgument)?;
            native.push(ez_gfx_backend_vulkan::NativeBufferBinding {
                allocation,
                offset: 0,
                range: 4,
                writable: requirement.writable,
            });
            native.push(ez_gfx_backend_vulkan::NativeBufferBinding {
                allocation,
                offset: COUNTER_BUFFER_ELEMENT_OFFSET,
                range: command_size,
                writable: requirement.writable,
            });
        } else {
            native.push(ez_gfx_backend_vulkan::NativeBufferBinding {
                allocation,
                offset: 0,
                range: *size,
                writable: requirement.writable,
            });
        }
    }
    Ok(native)
}

#[cfg(target_vendor = "apple")]
pub(super) fn metal_bindings<'a>(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    allocations: &'a HashMap<PackedHandle, (u64, NativeAllocation)>,
    vertex_heaps: &'a HashMap<String, GeometryAllocation>,
) -> std::result::Result<Vec<ez_gfx_backend_metal::native::NativeBufferBinding<'a>>, HalError> {
    let mut native = Vec::new();
    for requirement in layout.requirements() {
        if requirement.kind == ez_gfx_runtime::binding::BindingKind::VertexHeap {
            if requirement.descriptor_count != 1 {
                return Err(HalError::Unsupported);
            }
            let heap = vertex_heaps
                .get(&requirement.name)
                .ok_or(HalError::InvalidArgument)?;
            let NativeAllocation::Metal(allocation) = &heap.allocation else {
                return Err(HalError::InvalidArgument);
            };
            native.push(ez_gfx_backend_metal::native::NativeBufferBinding {
                allocation,
                offset: 0,
                index: requirement.binding as usize,
            });
            continue;
        }
        let public = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(HalError::InvalidArgument)?;
        let (handle, count) = match public.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Buffer(handle)
                if requirement.kind == ez_gfx_runtime::binding::BindingKind::Buffer
                    && requirement.descriptor_count == 1 =>
            {
                (handle.packed(), 1)
            }
            ez_gfx_runtime::binding::ResourceIdentity::Counter(handle)
                if requirement.kind == ez_gfx_runtime::binding::BindingKind::CounterBuffer
                    && requirement.descriptor_count == 2 =>
            {
                (handle.packed(), 2)
            }
            _ => return Err(HalError::Unsupported),
        };
        let (_, allocation) = allocations.get(&handle).ok_or(HalError::InvalidArgument)?;
        let NativeAllocation::Metal(allocation) = allocation else {
            return Err(HalError::InvalidArgument);
        };
        for descriptor in 0..count {
            let offset = if descriptor == 0 {
                0
            } else {
                usize::try_from(COUNTER_BUFFER_ELEMENT_OFFSET)
                    .map_err(|_| HalError::InvalidArgument)?
            };
            native.push(ez_gfx_backend_metal::native::NativeBufferBinding {
                allocation,
                offset,
                index: requirement.binding as usize + descriptor,
            });
        }
    }
    Ok(native)
}

#[cfg(windows)]
pub(super) fn dx12_bindings<'a>(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    allocations: &'a HashMap<PackedHandle, (u64, NativeAllocation)>,
    vertex_heaps: &'a HashMap<String, GeometryAllocation>,
) -> std::result::Result<Vec<ez_gfx_backend_dx12::native::NativeBufferBinding<'a>>, HalError> {
    let mut native = Vec::new();
    for requirement in layout.requirements() {
        if requirement.kind == ez_gfx_runtime::binding::BindingKind::VertexHeap {
            if requirement.descriptor_count != 1 {
                return Err(HalError::Unsupported);
            }
            let heap = vertex_heaps
                .get(&requirement.name)
                .ok_or(HalError::InvalidArgument)?;
            let NativeAllocation::Dx12(allocation) = &heap.allocation else {
                return Err(HalError::InvalidArgument);
            };
            native.push(ez_gfx_backend_dx12::native::NativeBufferBinding {
                allocation,
                offset: 0,
                writable: requirement.writable,
            });
            continue;
        }
        let public = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(HalError::InvalidArgument)?;
        let (handle, count) = match public.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Buffer(handle)
                if requirement.kind == ez_gfx_runtime::binding::BindingKind::Buffer
                    && requirement.descriptor_count == 1 =>
            {
                (handle.packed(), 1)
            }
            ez_gfx_runtime::binding::ResourceIdentity::Counter(handle)
                if requirement.kind == ez_gfx_runtime::binding::BindingKind::CounterBuffer
                    && requirement.descriptor_count == 2 =>
            {
                (handle.packed(), 2)
            }
            _ => return Err(HalError::Unsupported),
        };
        let (_, allocation) = allocations.get(&handle).ok_or(HalError::InvalidArgument)?;
        let NativeAllocation::Dx12(allocation) = allocation else {
            return Err(HalError::InvalidArgument);
        };
        for descriptor in 0..count {
            native.push(ez_gfx_backend_dx12::native::NativeBufferBinding {
                allocation,
                offset: if descriptor == 0 {
                    0
                } else {
                    COUNTER_BUFFER_ELEMENT_OFFSET
                },
                writable: requirement.writable,
            });
        }
    }
    Ok(native)
}

pub(super) fn wait_native_idle(context: &mut NativeContext) -> std::result::Result<(), HalError> {
    match context {
        NativeContext::Vulkan(context) => context.wait_idle(),
        #[cfg(windows)]
        NativeContext::Dx12(context) => context.wait_idle(),
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.wait_idle(),
    }
}

/// Reaps completed frame slots without blocking so polling texture paths observe
/// finished GPU work. Vulkan/Metal track submission in software slot state that
/// otherwise clears only on wrap-around or `wait_idle`; DX12 already consults
/// the live graphics fence on every gate check.
pub(super) fn poll_native_frame_completion(context: &mut NativeContext) -> crate::Result<()> {
    match context {
        NativeContext::Vulkan(context) => {
            context.poll_frame_completion().map_err(map_allocation)?;
        }
        #[cfg(windows)]
        NativeContext::Dx12(context) => context.poll_frame_completion().map_err(map_hal)?,
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.poll_frame_completion(),
    }
    Ok(())
}
pub(super) fn completed_native_frame_value(context: &mut NativeContext) -> crate::Result<u64> {
    match context {
        NativeContext::Vulkan(context) => context.completed_frame_value().map_err(map_allocation),
        #[cfg(windows)]
        NativeContext::Dx12(context) => context.completed_frame_value().map_err(map_allocation),
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.completed_frame_value().map_err(map_allocation),
    }
}

pub(super) fn last_native_frame_completion(
    context: &NativeContext,
) -> crate::Result<CompletionToken> {
    match context {
        NativeContext::Vulkan(context) => context.last_frame_completion(),
        #[cfg(windows)]
        NativeContext::Dx12(context) => context.last_frame_completion(),
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.last_frame_completion(),
    }
    .ok_or(Error::NativeFailure)
}

pub(super) fn allocate_native(
    context: &mut NativeContext,
    request: AllocationRequest,
) -> std::result::Result<NativeAllocation, ez_gfx_hal::AllocationError> {
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
) -> std::result::Result<(), ez_gfx_hal::AllocationError> {
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
) -> std::result::Result<CompletionToken, ez_gfx_hal::AllocationError> {
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
) -> std::result::Result<u64, ez_gfx_hal::AllocationError> {
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
) -> std::result::Result<u64, ez_gfx_hal::AllocationError> {
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
) -> std::result::Result<(), ez_gfx_hal::AllocationError> {
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

pub(super) fn retire_native_allocation(
    context: &mut NativeContext,
    allocation: NativeAllocation,
    completion: CompletionToken,
) -> std::result::Result<(), ez_gfx_hal::AllocationError> {
    match (context, allocation) {
        (NativeContext::Vulkan(context), NativeAllocation::Vulkan(allocation)) => {
            context.retire(allocation, completion)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeAllocation::Dx12(allocation)) => {
            context.retire(allocation, completion)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeAllocation::Metal(allocation)) => {
            context.retire(allocation, completion)
        }
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}
pub(super) fn map_allocation(error: ez_gfx_hal::AllocationError) -> Error {
    match error {
        ez_gfx_hal::AllocationError::ZeroSize
        | ez_gfx_hal::AllocationError::InvalidAlignment
        | ez_gfx_hal::AllocationError::NotHostVisible
        | ez_gfx_hal::AllocationError::InvalidAliasClass => Error::InvalidArgument,
        ez_gfx_hal::AllocationError::DeviceLost => Error::DeviceLost,
        ez_gfx_hal::AllocationError::Unsupported => Error::Unsupported,
        ez_gfx_hal::AllocationError::OutOfMemory | ez_gfx_hal::AllocationError::NativeFailure => {
            Error::NativeFailure
        }
    }
}

pub(super) fn map_geometry(error: GeometryError) -> Error {
    match error {
        GeometryError::StagingPoolExhausted => Error::NotReady,
        _ => Error::InvalidArgument,
    }
}

pub(super) fn map_lifecycle(error: LifecycleError) -> Error {
    match error {
        LifecycleError::DeviceLost | LifecycleError::AlreadyLost => Error::DeviceLost,
        error => Error::Lifecycle(error),
    }
}
pub(super) fn map_native_loss(identity: &ContextIdentity, error: HalError) -> Error {
    if error == HalError::DeviceLost {
        let _ = identity.mark_lost();
    }
    map_hal(error)
}
pub(super) fn result_status(result: crate::Result<()>) -> crate::Result<()> {
    result
}
pub(super) fn map_hal(error: HalError) -> Error {
    match error {
        HalError::InvalidArgument => Error::InvalidArgument,
        HalError::Unsupported => Error::Unsupported,
        HalError::NotReady => Error::NotReady,
        HalError::DeviceLost => Error::DeviceLost,
        HalError::OutOfMemory | HalError::NativeFailure => Error::NativeFailure,
    }
}
