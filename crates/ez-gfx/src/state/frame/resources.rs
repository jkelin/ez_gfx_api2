fn intern_buffer_resource(context: &mut ContextState, handle: PackedHandle) -> Result<ResourceId> {
    if let Some(resource) = context.frame_resources.get(&handle) {
        return Ok(*resource);
    }
    let size = context
        .allocations
        .get(&handle)
        .map(|(size, _)| *size)
        .ok_or(Error::InvalidContext)?;
    let desc = ResourceDesc::buffer(size, 4, ResourceLifetime::External)
        .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let initial = ResourceState::new(
        QueueKind::Transfer,
        ShaderStage::None,
        ResourceAccess::TransferWrite,
    )
    .map_err(|_| Error::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, initial)
        .map_err(|error| map_frame(&error))?;
    if let Some(ready) = context.allocation_ready.get(&handle).copied() {
        context
            .frame
            .set_resource_ready(resource, ready)
            .map_err(|error| map_frame(&error))?;
    }
    context.frame_resources.insert(handle, resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Buffer(handle));
    Ok(resource)
}

fn intern_surface_resource(context: &mut ContextState) -> Result<ResourceId> {
    if let Some(resource) = context.frame_surface {
        return Ok(resource);
    }
    let surface = context.active_surface.ok_or(Error::NotReady)?;
    let (width, height) = context
        .surfaces
        .get(&surface)
        .and_then(|surface| surface.state.extent())
        .ok_or(Error::NotReady)?;
    let desc = ResourceDesc::image(
        width,
        height,
        1,
        1,
        Format::Bgra8Srgb,
        1,
        ResourceLifetime::External,
    )
    .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let present = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::None,
        ResourceAccess::Present,
    )
    .map_err(|_| Error::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, present)
        .map_err(|error| map_frame(&error))?;
    context.frame_surface = Some(resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Surface(surface));
    Ok(resource)
}

fn intern_depth_resource(context: &mut ContextState) -> Result<ResourceId> {
    if let Some(resource) = context.frame_depth {
        return Ok(resource);
    }
    let surface = context.active_surface.ok_or(Error::NotReady)?;
    let (width, height) = context
        .surfaces
        .get(&surface)
        .and_then(|surface| surface.state.extent())
        .ok_or(Error::NotReady)?;
    let desc = ResourceDesc::image(
        width,
        height,
        1,
        1,
        Format::Depth32Float,
        1,
        ResourceLifetime::Transient,
    )
    .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    context.frame_depth = Some(resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Depth);
    Ok(resource)
}

fn intern_index_resource(context: &mut ContextState) -> Result<ResourceId> {
    if let Some(resource) = context.frame_index {
        return Ok(resource);
    }
    let heap = context.index_heap.as_ref().ok_or(Error::NotReady)?;
    let size = heap.size;
    let desc = ResourceDesc::buffer(size, 4, ResourceLifetime::External)
        .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let initial = ResourceState::new(
        QueueKind::Transfer,
        ShaderStage::None,
        ResourceAccess::TransferWrite,
    )
    .map_err(|_| Error::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, initial)
        .map_err(|error| map_frame(&error))?;
    if let Some(ready) = heap.ready {
        context
            .frame
            .set_resource_ready(resource, ready)
            .map_err(|error| map_frame(&error))?;
    }
    context.frame_index = Some(resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Index);
    Ok(resource)
}

fn intern_vertex_heap_resource(
    context: &mut ContextState,
    name: &str,
) -> Result<(ResourceId, u64)> {
    let heap = context.vertex_heaps.get(name).ok_or(Error::NotReady)?;
    let heap_id = heap.heap_id.ok_or(Error::NativeFailure)?;
    if let Some(resource) = context.frame_vertex_heaps.get(&heap_id) {
        return Ok((*resource, heap.size));
    }
    let size = heap.size;
    let ready = heap.ready;
    let desc = ResourceDesc::buffer(size, 16, ResourceLifetime::External)
        .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let initial = ResourceState::new(
        QueueKind::Transfer,
        ShaderStage::None,
        ResourceAccess::TransferWrite,
    )
    .map_err(|_| Error::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, initial)
        .map_err(|error| map_frame(&error))?;
    if let Some(ready) = ready {
        context
            .frame
            .set_resource_ready(resource, ready)
            .map_err(|error| map_frame(&error))?;
    }
    context.frame_vertex_heaps.insert(heap_id, resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::VertexHeap(heap_id));
    Ok((resource, size))
}

fn add_binding_accesses(
    context: &mut ContextState,
    mut node: NodeDesc,
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: binding::BindingProjection<'_>,
    queue: QueueKind,
    stage: ShaderStage,
    combined_indirect: Option<CounterBufferHandle>,
) -> Result<NodeDesc> {
    for requirement in layout.requirements() {
        if requirement.kind == ez_gfx_runtime::binding::BindingKind::VertexHeap {
            let (resource, size) = intern_vertex_heap_resource(context, &requirement.name)?;
            let access_state = ResourceState::new(
                queue,
                stage,
                if requirement.writable {
                    ResourceAccess::StorageReadWrite
                } else {
                    ResourceAccess::StorageRead
                },
            )
            .map_err(|_| Error::InvalidArgument)?;
            node = node.access(Access::buffer(
                resource,
                BufferRange::new(0, size).map_err(|_| Error::InvalidArgument)?,
                access_state,
            ));
            continue;
        }
        let binding = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(Error::InvalidArgument)?;
        let handle = match binding.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Buffer(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::Counter(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(_) => {
                return Err(Error::Unsupported);
            }
        };
        if combined_indirect.is_some_and(|indirect| indirect.packed() == handle) {
            continue;
        }
        let size = context
            .allocations
            .get(&handle)
            .map(|(size, _)| *size)
            .ok_or(Error::InvalidContext)?;
        let resource = intern_buffer_resource(context, handle)?;
        let writable = requirement.writable;
        let access_state = ResourceState::new(
            queue,
            stage,
            if writable {
                ResourceAccess::StorageReadWrite
            } else {
                ResourceAccess::StorageRead
            },
        )
        .map_err(|_| Error::InvalidArgument)?;
        node = node.access(Access::buffer(
            resource,
            BufferRange::new(0, size).map_err(|_| Error::InvalidArgument)?,
            access_state,
        ));
    }
    Ok(node)
}

fn intern_texture_resource(
    context: &mut ContextState,
    texture: TextureHandle,
) -> Result<ResourceId> {
    if let Some(resource) = context.frame_resources.get(&texture.packed()) {
        return Ok(*resource);
    }
    let (_, _, width, height, _) = context
        .textures
        .get(&texture)
        .ok_or(Error::InvalidContext)?;
    let desc = ResourceDesc::image(
        *width,
        *height,
        1,
        1,
        Format::Rgba8Unorm,
        1,
        ResourceLifetime::External,
    )
    .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let sampled = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    )
    .map_err(|_| Error::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, sampled)
        .map_err(|error| map_frame(&error))?;
    if let Some(ready) = context.texture_ready.get(&texture).copied() {
        context
            .frame
            .set_resource_ready(resource, ready)
            .map_err(|error| map_frame(&error))?;
    }
    context.frame_resources.insert(texture.packed(), resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Texture(texture));
    Ok(resource)
}
