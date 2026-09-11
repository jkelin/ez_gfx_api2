use core::ops::Range;

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

#[derive(Clone, Copy)]
pub(super) enum FrameBindingResource {
    Allocation(PackedHandle),
    VertexHeap(u32),
}

#[derive(Clone, Copy)]
pub(super) struct FrameBufferBindingRecord {
    pub(super) resource: FrameBindingResource,
    pub(super) offset: u64,
    pub(super) range: u64,
    pub(super) writable: bool,
    #[cfg_attr(
        not(target_vendor = "apple"),
        allow(dead_code, reason = "Metal alone consumes the reflected binding index")
    )]
    pub(super) index: usize,
}

pub(super) fn frame_bindings(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::ResourceIdentity],
    allocations: &HashMap<PackedHandle, (u64, NativeAllocation)>,
    vertex_heaps: &HashMap<String, GeometryAllocation>,
    scratch: &mut Vec<FrameBufferBindingRecord>,
) -> std::result::Result<Range<usize>, HalError> {
    append_frame_bindings(
        layout,
        bindings,
        |handle| allocations.get(&handle).map(|(size, _)| *size),
        |name| {
            vertex_heaps
                .get(name)
                .and_then(|heap| heap.heap_id.map(|id| (id, heap.size)))
        },
        scratch,
    )
}

fn append_frame_bindings(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::ResourceIdentity],
    mut allocation_size: impl FnMut(PackedHandle) -> Option<u64>,
    mut vertex_heap: impl FnMut(&str) -> Option<(u32, u64)>,
    scratch: &mut Vec<FrameBufferBindingRecord>,
) -> std::result::Result<Range<usize>, HalError> {
    let start = scratch.len();
    scratch.reserve(layout.requirements().len().saturating_mul(2));
    let mut bindings = bindings.iter();
    for requirement in layout.requirements() {
        if requirement.kind == ez_gfx_runtime::binding::BindingKind::VertexHeap {
            if requirement.descriptor_count != 1 {
                return Err(HalError::Unsupported);
            }
            let (heap_id, size) =
                vertex_heap(&requirement.name).ok_or(HalError::InvalidArgument)?;
            scratch.push(FrameBufferBindingRecord {
                resource: FrameBindingResource::VertexHeap(heap_id),
                offset: 0,
                range: size,
                writable: requirement.writable,
                index: requirement.binding as usize,
            });
            continue;
        }
        let resource = *bindings.next().ok_or(HalError::InvalidArgument)?;
        let handle = match resource {
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
        let size = allocation_size(handle).ok_or(HalError::InvalidArgument)?;
        for descriptor in 0..requirement.descriptor_count {
            let offset = if descriptor == 0 {
                0
            } else {
                COUNTER_BUFFER_ELEMENT_OFFSET
            };
            let range = if requirement.descriptor_count == 2 {
                if descriptor == 0 {
                    4
                } else {
                    size.checked_sub(offset).ok_or(HalError::InvalidArgument)?
                }
            } else {
                size
            };
            scratch.push(FrameBufferBindingRecord {
                resource: FrameBindingResource::Allocation(handle),
                offset,
                range,
                writable: requirement.writable,
                index: requirement.binding as usize + descriptor as usize,
            });
        }
    }
    if bindings.next().is_some() {
        return Err(HalError::InvalidArgument);
    }
    Ok(start..scratch.len())
}

pub(super) fn prepare_frame_binding_scratch(
    payloads: &[super::ExecutableNode],
    binding_resources: &[ez_gfx_runtime::binding::ResourceIdentity],
    allocations: &HashMap<PackedHandle, (u64, NativeAllocation)>,
    vertex_heaps: &HashMap<String, GeometryAllocation>,
    scratch: &mut Vec<FrameBufferBindingRecord>,
    ranges: &mut Vec<Range<usize>>,
) -> std::result::Result<(), HalError> {
    scratch.clear();
    ranges.clear();
    ranges.reserve(payloads.len());
    for payload in payloads {
        let range = match payload {
            super::ExecutableNode::Graphics {
                layout, bindings, ..
            }
            | super::ExecutableNode::Compute {
                layout, bindings, ..
            } => frame_bindings(
                layout,
                binding_resources
                    .get(bindings.clone())
                    .ok_or(HalError::InvalidArgument)?,
                allocations,
                vertex_heaps,
                scratch,
            )?,
            super::ExecutableNode::TextureReadback { .. }
            | super::ExecutableNode::RenderTargetReadback { .. }
            | super::ExecutableNode::Present { .. } => scratch.len()..scratch.len(),
        };
        ranges.push(range);
    }
    Ok(())
}

pub(super) struct FrameBindingSource<'a> {
    pub(super) records: &'a [FrameBufferBindingRecord],
    pub(super) allocations: &'a HashMap<PackedHandle, (u64, NativeAllocation)>,
    pub(super) vertex_heaps: &'a HashMap<String, GeometryAllocation>,
}

impl FrameBindingSource<'_> {
    fn allocation(
        &self,
        resource: FrameBindingResource,
    ) -> std::result::Result<&NativeAllocation, HalError> {
        match resource {
            FrameBindingResource::Allocation(handle) => self
                .allocations
                .get(&handle)
                .map(|(_, allocation)| allocation)
                .ok_or(HalError::InvalidArgument),
            FrameBindingResource::VertexHeap(heap_id) => self
                .vertex_heaps
                .values()
                .find(|heap| heap.heap_id == Some(heap_id))
                .map(|heap| &heap.allocation)
                .ok_or(HalError::InvalidArgument),
        }
    }
}

impl ez_gfx_backend_vulkan::NativeBufferBindingSource for FrameBindingSource<'_> {
    fn len(&self) -> usize {
        self.records.len()
    }

    fn visit(
        &self,
        visitor: &mut dyn FnMut(
            usize,
            &ez_gfx_backend_vulkan::NativeBufferBinding<'_>,
        ) -> std::result::Result<(), HalError>,
    ) -> std::result::Result<(), HalError> {
        for (index, record) in self.records.iter().enumerate() {
            #[cfg(not(any(windows, target_vendor = "apple")))]
            let NativeAllocation::Vulkan(allocation) = self.allocation(record.resource)?;
            #[cfg(any(windows, target_vendor = "apple"))]
            let NativeAllocation::Vulkan(allocation) = self.allocation(record.resource)? else {
                return Err(HalError::InvalidArgument);
            };
            visitor(
                index,
                &ez_gfx_backend_vulkan::NativeBufferBinding {
                    allocation,
                    offset: record.offset,
                    range: record.range,
                    writable: record.writable,
                },
            )?;
        }
        Ok(())
    }
}

#[cfg(windows)]
impl ez_gfx_backend_dx12::native::NativeBufferBindingSource for FrameBindingSource<'_> {
    fn len(&self) -> usize {
        self.records.len()
    }

    fn visit(
        &self,
        visitor: &mut dyn FnMut(
            usize,
            &ez_gfx_backend_dx12::native::NativeBufferBinding<'_>,
        ) -> std::result::Result<(), HalError>,
    ) -> std::result::Result<(), HalError> {
        for (index, record) in self.records.iter().enumerate() {
            let NativeAllocation::Dx12(allocation) = self.allocation(record.resource)? else {
                return Err(HalError::InvalidArgument);
            };
            visitor(
                index,
                &ez_gfx_backend_dx12::native::NativeBufferBinding {
                    allocation,
                    offset: record.offset,
                    writable: record.writable,
                },
            )?;
        }
        Ok(())
    }
}

#[cfg(target_vendor = "apple")]
impl ez_gfx_backend_metal::native::NativeBufferBindingSource for FrameBindingSource<'_> {
    fn len(&self) -> usize {
        self.records.len()
    }

    fn visit(
        &self,
        visitor: &mut dyn FnMut(
            usize,
            &ez_gfx_backend_metal::native::NativeBufferBinding<'_>,
        ) -> std::result::Result<(), HalError>,
    ) -> std::result::Result<(), HalError> {
        for (index, record) in self.records.iter().enumerate() {
            let NativeAllocation::Metal(allocation) = self.allocation(record.resource)? else {
                return Err(HalError::InvalidArgument);
            };
            visitor(
                index,
                &ez_gfx_backend_metal::native::NativeBufferBinding {
                    allocation,
                    offset: usize::try_from(record.offset)
                        .map_err(|_| HalError::InvalidArgument)?,
                    index: record.index,
                },
            )?;
        }
        Ok(())
    }
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

pub(super) fn native_device_initialized(context: &NativeContext) -> bool {
    match context {
        NativeContext::Vulkan(context) => context.adapter_info().is_some(),
        #[cfg(windows)]
        NativeContext::Dx12(_) => true,
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(_) => true,
    }
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

#[cfg(test)]
mod binding_scratch_tests {
    use super::append_frame_bindings;
    use crate::state::frame::BindingProjection;
    use ez_gfx_artifact::Stage;
    use ez_gfx_core::{
        Backend,
        handle::{BufferHandle, LocalHandle, PackedHandle, ShaderHandle},
    };
    use ez_gfx_hal::QueueKind;
    use ez_gfx_runtime::{
        binding::{PublicBinding, ReflectedBindings, ResourceIdentity},
        frame::{ExecutableNode, FrameRecorder},
        graph::NodeDesc,
    };
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    struct CountingAllocator;
    static ENABLED: AtomicBool = AtomicBool::new(false);
    static CALLS: AtomicUsize = AtomicUsize::new(0);

    // SAFETY: every allocation operation preserves `GlobalAlloc`'s pointer and layout contracts by
    // delegating unchanged requests to the system allocator.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: the unchanged request is delegated to the system allocator.
            let pointer = unsafe { System.alloc(layout) };
            if ENABLED.load(Ordering::Relaxed) && !pointer.is_null() {
                CALLS.fetch_add(1, Ordering::Relaxed);
            }
            pointer
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            // SAFETY: the pointer and layout came from the system allocator above.
            unsafe { System.dealloc(pointer, layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    #[test]
    fn more_than_64_bindings_are_accepted_without_warmed_allocations() {
        let parameters = (0..65)
            .map(|index| {
                format!(
                    r#"{{"semantic_name":"buffer{index}","api_kind":"buffer","binding_index":{index},"binding_space":0}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let metadata = format!(
            r#"{{"reflections":[{{"target":"Spirv","entry":"main","stage":"Compute","reflection":{{"parameters":[{parameters}]}}}}]}}"#
        );
        let layout =
            ReflectedBindings::parse(metadata.as_bytes(), Backend::Vulkan, "main", Stage::Compute)
                .unwrap();
        let public_bindings = (0..65)
            .map(|index| {
                let packed = PackedHandle::child(
                    LocalHandle::new(1, 1).unwrap(),
                    LocalHandle::new(index + 1, 1).unwrap(),
                )
                .unwrap();
                PublicBinding {
                    name: format!("buffer{index}"),
                    resource: ResourceIdentity::Buffer(BufferHandle::from_packed(packed).unwrap()),
                }
            })
            .collect::<Vec<_>>();
        let bindings = BindingProjection::new(&layout, &public_bindings)
            .resources()
            .collect::<Vec<_>>();
        let mut scratch = Vec::new();
        append_frame_bindings(&layout, &bindings, |_| Some(1024), |_| None, &mut scratch).unwrap();
        assert_eq!(scratch.len(), 65);

        CALLS.store(0, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);
        for _ in 0..500 {
            scratch.clear();
            append_frame_bindings(&layout, &bindings, |_| Some(1024), |_| None, &mut scratch)
                .unwrap();
        }
        ENABLED.store(false, Ordering::Relaxed);
        assert_eq!(CALLS.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn warmed_more_than_64_binding_execution_recording_performs_no_allocations() {
        let parameters = (0..65)
            .map(|index| {
                format!(
                    r#"{{"semantic_name":"buffer{index}","api_kind":"buffer","binding_index":{index},"binding_space":0}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let metadata = format!(
            r#"{{"reflections":[{{"target":"Spirv","entry":"main","stage":"Compute","reflection":{{"parameters":[{parameters}]}}}}]}}"#
        );
        let layout =
            ReflectedBindings::parse(metadata.as_bytes(), Backend::Vulkan, "main", Stage::Compute)
                .unwrap();
        let mut bindings = (0..65)
            .map(|index| PublicBinding {
                name: format!("buffer{index}"),
                resource: ResourceIdentity::Buffer(
                    BufferHandle::from_packed(
                        PackedHandle::child(
                            LocalHandle::new(1, 1).unwrap(),
                            LocalHandle::new(index + 1, 1).unwrap(),
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                ),
            })
            .collect::<Vec<_>>();
        let shader = ShaderHandle::from_packed(
            PackedHandle::child(
                LocalHandle::new(1, 1).unwrap(),
                LocalHandle::new(100, 1).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let mut recorder = FrameRecorder::new(1).unwrap();
        let baseline_retained = recorder.workspace_stats().retained_bytes;
        for _ in 0..4 {
            recorder.begin().unwrap();
            let projection = BindingProjection::new(&layout, &bindings);
            projection.validate().unwrap();
            let payload_layout = layout.clone();
            recorder
                .record_bound_node(
                    NodeDesc::new("compute", QueueKind::Compute),
                    projection.resources(),
                    move |bindings| ExecutableNode::Compute {
                        shader,
                        groups: [1, 1, 1],
                        bindings,
                        layout: payload_layout,
                    },
                )
                .unwrap();
            recorder.abort();
        }

        CALLS.store(0, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);
        for _ in 0..500 {
            recorder.begin().unwrap();
            let projection = BindingProjection::new(&layout, &bindings);
            projection.validate().unwrap();
            let payload_layout = layout.clone();
            recorder
                .record_bound_node(
                    NodeDesc::new("compute", QueueKind::Compute),
                    projection.resources(),
                    move |bindings| ExecutableNode::Compute {
                        shader,
                        groups: [1, 1, 1],
                        bindings,
                        layout: payload_layout,
                    },
                )
                .unwrap();
            recorder.abort();
        }
        ENABLED.store(false, Ordering::Relaxed);
        assert_eq!(CALLS.load(Ordering::Relaxed), 0);

        let original = bindings[0].resource;
        recorder.begin().unwrap();
        for _ in 0..1 {
            let projection = BindingProjection::new(&layout, &bindings);
            let payload_layout = layout.clone();
            recorder
                .record_bound_node(
                    NodeDesc::new("compute-before-rebind", QueueKind::Compute),
                    projection.resources(),
                    move |bindings| ExecutableNode::Compute {
                        shader,
                        groups: [1, 1, 1],
                        bindings,
                        layout: payload_layout,
                    },
                )
                .unwrap();
        }
        bindings[0].resource = ResourceIdentity::Buffer(
            BufferHandle::from_packed(
                PackedHandle::child(
                    LocalHandle::new(1, 1).unwrap(),
                    LocalHandle::new(99, 1).unwrap(),
                )
                .unwrap(),
            )
            .unwrap(),
        );
        let rebound = bindings[0].resource;
        let projection = BindingProjection::new(&layout, &bindings);
        let payload_layout = layout.clone();
        recorder
            .record_bound_node(
                NodeDesc::new("compute-after-rebind", QueueKind::Compute),
                projection.resources(),
                move |bindings| ExecutableNode::Compute {
                    shader,
                    groups: [1, 1, 1],
                    bindings,
                    layout: payload_layout,
                },
            )
            .unwrap();
        let submission = recorder.submit().unwrap();
        assert_eq!(submission.binding_resources[0], original);
        assert_eq!(submission.binding_resources[65], rebound);
        recorder.finish(submission).unwrap();
        assert!(recorder.workspace_stats().retained_bytes > baseline_retained);
    }
}
