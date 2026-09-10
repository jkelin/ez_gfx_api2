fn rollback_transient_internment(context: &mut ContextState) {
    for buffer in context.transient_buffers.values_mut() {
        if buffer.usage == super::TransientUse::Interned(context.frame_serial) {
            buffer.usage = super::TransientUse::Available;
        }
    }
}

fn invalidate_unsafe_transients(context: &mut ContextState) {
    let handles = context
        .transient_buffers
        .iter()
        .filter_map(|(handle, buffer)| {
            (buffer.usage == super::TransientUse::Interned(context.frame_serial)).then_some(*handle)
        })
        .collect::<Vec<_>>();
    for handle in handles {
        if let Ok(kind) = context.identity.resource_kind(handle) {
            let _ = context.identity.remove(handle, kind);
            if kind == ResourceKind::CounterBuffer
                && let Ok(typed) = CounterBufferHandle::from_packed(handle)
            {
                context.indirects.remove(&typed);
            }
        }
        context.transient_buffers.remove(&handle);
        // Native ownership is uncertain after a failed idle drain. Keep the
        // allocation quarantined in `allocations` for terminal context cleanup.
    }
}

#[cfg(test)]
mod capture_tests {
    use super::should_capture_presented;

    #[test]
    fn presented_capture_combines_persistent_and_one_frame_requests() {
        for (cache, request, expected) in [
            (false, false, false),
            (false, true, true),
            (true, false, true),
            (true, true, true),
        ] {
            assert_eq!(should_capture_presented(cache, request), expected);
        }
    }
}

fn recycle_consumed_transients(
    context: &mut ContextState,
    completion: ez_gfx_hal::CompletionToken,
) -> Result<()> {
    let handles = context
        .transient_buffers
        .iter()
        .filter_map(|(handle, buffer)| {
            (buffer.usage == super::TransientUse::Interned(context.frame_serial)).then_some(*handle)
        })
        .collect::<Vec<_>>();
    for handle in handles {
        let kind = context
            .identity
            .resource_kind(handle)
            .map_err(map_lifecycle)?;
        context
            .identity
            .remove(handle, kind)
            .map_err(map_lifecycle)?;
        let metadata = context
            .transient_buffers
            .remove(&handle)
            .ok_or(Error::InvalidContext)?;
        context.allocation_ready.remove(&handle);
        let (_, allocation) = context
            .allocations
            .remove(&handle)
            .ok_or(Error::InvalidContext)?;
        match kind {
            ResourceKind::Buffer => context
                .buffer_pool
                .entry(metadata.element_size)
                .or_insert_with(|| ez_gfx_hal::ReusableStagingPool::new(256))
                .put(metadata.byte_capacity, allocation, Some(completion)),
            ResourceKind::CounterBuffer => {
                let typed =
                    CounterBufferHandle::from_packed(handle).map_err(|_| Error::NativeFailure)?;
                context.indirects.remove(&typed);
                context
                    .counter_pool
                    .put(metadata.byte_capacity, allocation, Some(completion));
            }
            _ => return Err(Error::InvalidContext),
        }
    }
    Ok(())
}
