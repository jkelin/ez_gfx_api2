use crate::Result;

use super::{
    AllocationRequest, BufferHandle, COUNTER_BUFFER_ELEMENT_OFFSET, ContextHandle, ContextState,
    CounterBufferHandle, DrawIndexedCommand, Error, IndexedIndirectBuffer, MemoryClass,
    NativeAllocation, NativeContext, PackedHandle, ReclaimableStaging, ResourceKind,
    TransientBuffer, TransientUse, allocate_native, completed_native_frame_value,
    completed_texture_transfer_native, completed_transfer_native, free_native_allocation,
    largest_native_texture_staging, map_allocation, map_lifecycle, native_device_initialized,
    pop_largest_native_texture_staging, result_status, retained_native_texture_staging,
    retire_native_allocation, stage_upload, with_context_mut,
};

/// Acquires a runtime-typed structured buffer for the C ABI.
#[cfg(feature = "ffi")]
#[doc(hidden)]
pub fn acquire_buffer_raw(
    context: ContextHandle,
    element_size: u32,
    element_count: u32,
) -> Result<BufferHandle> {
    acquire_buffer_raw_impl(context, element_size, element_count)
}

pub(crate) fn acquire_buffer_sized(
    context: ContextHandle,
    element_size: u32,
    element_count: u32,
) -> Result<BufferHandle> {
    acquire_buffer_raw_impl(context, element_size, element_count)
}

fn acquire_buffer_raw_impl(
    context: ContextHandle,
    element_size: u32,
    element_count: u32,
) -> Result<BufferHandle> {
    let (_, size) = checked_element_range(element_size, element_count, 0, element_count)?;
    with_context_mut(context, |context| {
        require_recording(context)?;
        let completed = completed_native_frame_value(&mut context.native)?;
        let (reused, mut stale) = {
            let pool = context.buffer_pool.entry(element_size).or_insert_with(|| {
                // Lazily created stride pools inherit the finite per-entry budget;
                // creation alone never evicts, pressure trims below do.
                let mut pool = ez_gfx_hal::ReusableStagingPool::new(256);
                pool.set_byte_budget(ez_gfx_hal::DEFAULT_BUFFER_STAGING_BUDGET);
                pool
            });
            let reused = pool.take(size, ez_gfx_hal::QueueKind::Graphics, completed);
            let mut stale = pool.trim(ez_gfx_hal::QueueKind::Graphics, completed);
            // Idle trimming evicts only stale buckets; the budget trim evicts
            // completed-but-fresh buckets once retention exceeds the ceiling.
            stale.extend(pool.trim_to_budget(ez_gfx_hal::QueueKind::Graphics, completed));
            (reused, stale)
        };
        // Stride pools retain independently, so only this global pass bounds
        // their sum. The transfer counter is queried only under aggregate
        // pressure, never on the ordinary acquisition hot path.
        trim_staging_to_aggregate_budget_after_graphics(context, completed, &mut stale)?;
        observe_staging_high_water(context);
        for allocation in stale {
            free_native_allocation(&mut context.native, allocation).map_err(map_allocation)?;
        }
        let (byte_capacity, allocation) = if let Some(reused) = reused {
            reused
        } else {
            let request = AllocationRequest::new(size, 16, MemoryClass::Device, false, None)
                .map_err(|_| Error::InvalidArgument)?;
            (
                size,
                allocate_native(&mut context.native, request).map_err(map_allocation)?,
            )
        };
        insert_transient(
            context,
            ResourceKind::Buffer,
            size,
            allocation,
            TransientBuffer {
                element_size,
                element_count,
                byte_capacity,
                usage: TransientUse::Available,
            },
        )
        .and_then(|packed| BufferHandle::from_packed(packed).map_err(|_| Error::NativeFailure))
    })
}

/// Allocates a per-frame counter and indexed-command buffer.
///
/// The count occupies the first four bytes; commands begin at the shared aligned element offset.
///
/// # Errors
///
/// Returns an error outside frame recording, for invalid capacity, exhausted
/// handles, overflow, or native allocation failure.
pub fn acquire_counter(context: ContextHandle, capacity: u32) -> Result<CounterBufferHandle> {
    with_context_mut(context, |context| {
        require_recording(context)?;
        let buffer = IndexedIndirectBuffer::new(capacity).map_err(|_| Error::InvalidArgument)?;
        let (_, command_size) = checked_element_range(20, capacity, 0, capacity)?;
        let size = command_size
            .checked_add(COUNTER_BUFFER_ELEMENT_OFFSET)
            .ok_or(Error::InvalidArgument)?;
        let completed = completed_native_frame_value(&mut context.native)?;
        let reused = context
            .counter_pool
            .take(size, ez_gfx_hal::QueueKind::Graphics, completed);
        let mut stale = context
            .counter_pool
            .trim(ez_gfx_hal::QueueKind::Graphics, completed);
        stale.extend(
            context
                .counter_pool
                .trim_to_budget(ez_gfx_hal::QueueKind::Graphics, completed),
        );
        trim_staging_to_aggregate_budget_after_graphics(context, completed, &mut stale)?;
        observe_staging_high_water(context);
        for allocation in stale {
            free_native_allocation(&mut context.native, allocation).map_err(map_allocation)?;
        }
        let (byte_capacity, allocation) = if let Some(reused) = reused {
            reused
        } else {
            let request = AllocationRequest::new(size, 4, MemoryClass::Device, false, None)
                .map_err(|_| Error::InvalidArgument)?;
            (
                size,
                allocate_native(&mut context.native, request).map_err(map_allocation)?,
            )
        };
        let packed = insert_transient(
            context,
            ResourceKind::CounterBuffer,
            size,
            allocation,
            TransientBuffer {
                element_size: 20,
                element_count: capacity,
                byte_capacity,
                usage: TransientUse::Available,
            },
        )?;
        let typed = CounterBufferHandle::from_packed(packed).map_err(|_| Error::NativeFailure)?;
        context.indirects.insert(typed, buffer);
        Ok(typed)
    })
}

/// Writes and publishes a contiguous indexed-draw command range.
///
/// Active count becomes `max(previous_count, start_index + commands.len())`.
///
/// # Errors
///
/// Returns an error for a stale, consumed, foreign, or out-of-range handle,
/// checked arithmetic failure, or failed upload.
#[cfg_attr(
    not(any(test, feature = "ffi")),
    allow(
        dead_code,
        reason = "only raw FFI and state tests write typed commands"
    )
)]
pub fn write_counter_commands(
    context: ContextHandle,
    indirect: CounterBufferHandle,
    start_index: u32,
    commands: &[DrawIndexedCommand],
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let handle = indirect.packed();
        validate_writable_transient(context, handle, ResourceKind::CounterBuffer)?;
        let count = u32::try_from(commands.len()).map_err(|_| Error::InvalidArgument)?;
        context
            .indirects
            .get_mut(&indirect)
            .ok_or(Error::InvalidContext)?
            .write_batch(start_index, commands)
            .map_err(|_| Error::InvalidArgument)?;
        let visible_count = context
            .indirects
            .get(&indirect)
            .ok_or(Error::InvalidContext)?
            .draw_count();
        if commands.is_empty() {
            return Ok(());
        }
        let offset = u64::from(start_index)
            .checked_mul(20)
            .and_then(|offset| offset.checked_add(COUNTER_BUFFER_ELEMENT_OFFSET))
            .ok_or(Error::InvalidArgument)?;
        let byte_size = u64::from(count)
            .checked_mul(20)
            .ok_or(Error::InvalidArgument)?;
        if byte_size == 0 {
            return Ok(());
        }
        let byte_size = usize::try_from(byte_size).map_err(|_| Error::InvalidArgument)?;
        let ContextState {
            native,
            staging,
            transfer_pool,
            allocations,
            allocation_ready,
            counter_scratch,
            ..
        } = context;
        // Retained scratch keeps steady-state counter writes allocation-free; the
        // copy into mapped staging storage still occurs per call via stage_upload.
        counter_scratch.clear();
        counter_scratch.reserve(byte_size);
        for command in commands {
            counter_scratch.extend_from_slice(&command.index_count.to_le_bytes());
            counter_scratch.extend_from_slice(&command.instance_count.to_le_bytes());
            counter_scratch.extend_from_slice(&command.first_index.to_le_bytes());
            counter_scratch.extend_from_slice(&command.vertex_offset.to_le_bytes());
            counter_scratch.extend_from_slice(&command.first_instance.to_le_bytes());
        }
        // The fallible tail runs inside a closure so a failed upload still
        // trims below: without the guard, one failed multi-megabyte write
        // would pin its capacity for the context lifetime.
        let result = (|| -> Result<()> {
            let (_, allocation) = allocations.get(&handle).ok_or(Error::InvalidContext)?;
            stage_upload(
                native,
                staging,
                transfer_pool,
                allocation,
                0,
                &visible_count.to_le_bytes(),
            )
            .map_err(map_allocation)?;
            let token = stage_upload(
                native,
                staging,
                transfer_pool,
                allocation,
                offset,
                counter_scratch,
            )
            .map_err(map_allocation)?;
            allocation_ready.insert(handle, token);
            Ok(())
        })();
        trim_counter_scratch(counter_scratch);
        observe_staging_high_water(context);
        result
    }))
}

/// Maximum counter serialization capacity retained across writes.
///
/// Steady-state counter payloads are a few kilobytes; one pathological
/// multi-megabyte write must not pin that capacity for the context lifetime.
const COUNTER_SCRATCH_RETAIN_LIMIT: usize = 64 * 1024;

/// Releases retained serialization capacity above the retention limit.
///
/// Call after the staged payload no longer needs scratch storage; steady-state
/// writes below the limit keep reusing capacity allocation-free.
fn trim_counter_scratch(scratch: &mut Vec<u8>) {
    // Clearing first means the shrink frees everything above zero instead of
    // pinning the just-written payload size.
    scratch.clear();
    if scratch.capacity() > COUNTER_SCRATCH_RETAIN_LIMIT {
        scratch.shrink_to_fit();
    }
}
fn counter_payload<'scratch>(
    scratch: &'scratch mut Vec<u8>,
    bytes: &[u8],
    initial_count: u32,
) -> Result<&'scratch [u8]> {
    // Padding is explicitly zeroed so no stale pooled bytes exist between the count and elements.
    let element_offset =
        usize::try_from(COUNTER_BUFFER_ELEMENT_OFFSET).map_err(|_| Error::InvalidArgument)?;
    let payload_size = bytes
        .len()
        .checked_add(element_offset)
        .ok_or(Error::InvalidArgument)?;
    // Retained scratch removes the per-call payload Vec; capacity persists across writes.
    scratch.clear();
    scratch.reserve(payload_size);
    scratch.extend_from_slice(&initial_count.to_le_bytes());
    scratch.resize(element_offset, 0);
    scratch.extend_from_slice(bytes);
    Ok(scratch)
}

/// Stages a complete counter-buffer payload without an intermediate command copy.
///
/// # Errors
///
/// Returns an error for stale, consumed, foreign, incorrectly sized data, or an
/// initial count exceeding capacity.
#[doc(hidden)]
pub fn write_counter_bytes(
    context: ContextHandle,
    counter: CounterBufferHandle,
    bytes: &[u8],
    initial_count: u32,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let handle = counter.packed();
        validate_writable_transient(context, handle, ResourceKind::CounterBuffer)?;
        let metadata = context
            .transient_buffers
            .get(&handle)
            .copied()
            .ok_or(Error::InvalidContext)?;
        let expected = usize::try_from(metadata.element_count)
            .ok()
            .and_then(|count| count.checked_mul(metadata.element_size as usize))
            .ok_or(Error::InvalidArgument)?;
        // Partial command initialization would leave unspecified command bytes visible.
        if metadata.element_size != 20
            || bytes.len() != expected
            || initial_count > metadata.element_count
        {
            return Err(Error::InvalidArgument);
        }
        let ContextState {
            native,
            staging,
            transfer_pool,
            allocations,
            allocation_ready,
            counter_scratch,
            ..
        } = context;
        // Same retained scratch as the command path; the guard trims even when
        // payload construction or the upload below fails partway through.
        let result = (|| -> Result<()> {
            let payload = counter_payload(counter_scratch, bytes, initial_count)?;
            let (_, allocation) = allocations.get(&handle).ok_or(Error::InvalidContext)?;
            let token = stage_upload(native, staging, transfer_pool, allocation, 0, payload)
                .map_err(map_allocation)?;
            allocation_ready.insert(handle, token);
            Ok(())
        })();
        trim_counter_scratch(counter_scratch);
        observe_staging_high_water(context);
        result
    }))
}

/// Writes runtime-typed buffer elements for the C ABI.
#[cfg(feature = "ffi")]
#[doc(hidden)]
pub fn write_buffer_raw(
    context: ContextHandle,
    buffer: BufferHandle,
    start_index: u32,
    element_count: u32,
    element_size: u32,
    bytes: &[u8],
) -> Result<()> {
    let value_count = usize::try_from(element_count).map_err(|_| Error::InvalidArgument)?;
    write_buffer_raw_impl(
        context,
        buffer,
        start_index,
        element_size,
        bytes,
        value_count,
    )
}

pub(crate) fn write_buffer_bytes(
    context: ContextHandle,
    buffer: BufferHandle,
    element_size: u32,
    bytes: &[u8],
) -> Result<()> {
    let value_count = bytes
        .len()
        .checked_div(element_size as usize)
        .ok_or(Error::InvalidArgument)?;
    write_buffer_raw_impl(context, buffer, 0, element_size, bytes, value_count)
}

fn write_buffer_raw_impl(
    context: ContextHandle,
    buffer: BufferHandle,
    start_index: u32,
    element_size: u32,
    bytes: &[u8],
    value_count: usize,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let handle = buffer.packed();
        validate_writable_transient(context, handle, ResourceKind::Buffer)?;
        let metadata = *context
            .transient_buffers
            .get(&handle)
            .ok_or(Error::InvalidContext)?;
        if element_size == 0 || metadata.element_size != element_size {
            return Err(Error::InvalidArgument);
        }
        let count = u32::try_from(value_count).map_err(|_| Error::InvalidArgument)?;
        let (offset, byte_size) = checked_element_range(
            metadata.element_size,
            metadata.element_count,
            start_index,
            count,
        )?;
        if usize::try_from(byte_size).ok() != Some(bytes.len()) {
            return Err(Error::InvalidArgument);
        }
        if byte_size == 0 {
            return Ok(());
        }
        let ContextState {
            native,
            staging,
            transfer_pool,
            allocations,
            allocation_ready,
            ..
        } = context;
        let (_, allocation) = allocations.get(&handle).ok_or(Error::InvalidContext)?;
        let result = stage_upload(native, staging, transfer_pool, allocation, offset, bytes)
            .map_err(map_allocation);
        if let Ok(token) = &result {
            allocation_ready.insert(handle, *token);
        }
        observe_staging_high_water(context);
        result.map(|_| ())
    }))
}

/// Releases a counter buffer that was not consumed by a recorded frame.
pub fn release_counter(context: ContextHandle, counter: CounterBufferHandle) {
    let _ = release_transient(
        context,
        counter.packed(),
        ResourceKind::CounterBuffer,
        Some(counter),
    );
}

/// Releases a buffer that was not consumed by a recorded frame.
pub fn release_buffer(context: ContextHandle, buffer: BufferHandle) {
    let _ = release_transient(context, buffer.packed(), ResourceKind::Buffer, None);
}

fn insert_transient(
    context: &mut ContextState,
    kind: ResourceKind,
    size: u64,
    allocation: NativeAllocation,
    metadata: TransientBuffer,
) -> Result<PackedHandle> {
    let handle = match context.identity.insert(kind) {
        Ok(handle) => handle,
        Err(error) => {
            let _ = free_native_allocation(&mut context.native, allocation);
            return Err(map_lifecycle(error));
        }
    };
    context.allocations.insert(handle, (size, allocation));
    context.transient_buffers.insert(handle, metadata);
    Ok(handle)
}

fn release_transient(
    context_handle: ContextHandle,
    handle: PackedHandle,
    kind: ResourceKind,
    indirect: Option<CounterBufferHandle>,
) -> Result<()> {
    with_context_mut(context_handle, |context| {
        context
            .identity
            .resolve(handle, kind)
            .map_err(map_lifecycle)?;
        let metadata = context
            .transient_buffers
            .get(&handle)
            .copied()
            .ok_or(Error::InvalidContext)?;
        if metadata.usage != TransientUse::Available {
            return Err(Error::NotReady);
        }
        context
            .identity
            .remove(handle, kind)
            .map_err(map_lifecycle)?;
        context.transient_buffers.remove(&handle);
        if let Some(indirect) = indirect {
            context.indirects.remove(&indirect);
        }
        let ready = context.allocation_ready.remove(&handle);
        let (_, allocation) = context
            .allocations
            .remove(&handle)
            .ok_or(Error::InvalidContext)?;
        match ready {
            Some(completion) => {
                retire_native_allocation(&mut context.native, allocation, completion)
            }
            None => free_native_allocation(&mut context.native, allocation),
        }
        .map_err(map_allocation)
    })
}

// Unrecorded writes may still be pending on transfer; retire those allocations
// against their upload token instead of admitting them to a graphics-completion pool.
pub(super) fn reclaim_available_transients(context: &mut ContextState) -> Result<()> {
    let handles = context
        .transient_buffers
        .iter()
        .filter_map(|(handle, metadata)| {
            (metadata.usage == TransientUse::Available).then_some(*handle)
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
        context.transient_buffers.remove(&handle);
        if kind == ResourceKind::CounterBuffer
            && let Ok(indirect) = CounterBufferHandle::from_packed(handle)
        {
            context.indirects.remove(&indirect);
        }
        let ready = context.allocation_ready.remove(&handle);
        let (_, allocation) = context
            .allocations
            .remove(&handle)
            .ok_or(Error::InvalidContext)?;
        match ready {
            Some(completion) => {
                retire_native_allocation(&mut context.native, allocation, completion)
            }
            None => free_native_allocation(&mut context.native, allocation),
        }
        .map_err(map_allocation)?;
    }

    Ok(())
}

/// Evicts completed staging buckets down to finite budgets, freeing them natively.
///
/// Shared by the memory-pressure entry point and the native-idle path; both run
/// outside frame recording, where completion values are fresh. Only buckets
/// retired by completed GPU work are evicted: in-flight retention may keep a
/// pool over budget, which the next idle or pressure call revisits.
pub(super) fn trim_staging_caches(context: &mut ContextState) -> Result<()> {
    let completions = if native_device_initialized(&context.native) {
        staging_completions(context)?
    } else {
        // No queue exists before Vulkan device admission, so no submitted
        // retirement token can exist; unsubmitted buckets remain reclaimable.
        StagingCompletions {
            transfer: 0,
            graphics: 0,
        }
    };
    let mut stale = context
        .staging
        .trim(ez_gfx_hal::QueueKind::Transfer, completions.transfer);
    stale.extend(
        context
            .staging
            .trim_to_budget(ez_gfx_hal::QueueKind::Transfer, completions.transfer),
    );
    for pool in context.buffer_pool.values_mut() {
        stale.extend(pool.trim(ez_gfx_hal::QueueKind::Graphics, completions.graphics));
        stale.extend(pool.trim_to_budget(ez_gfx_hal::QueueKind::Graphics, completions.graphics));
    }
    stale.extend(
        context
            .counter_pool
            .trim(ez_gfx_hal::QueueKind::Graphics, completions.graphics),
    );
    stale.extend(
        context
            .counter_pool
            .trim_to_budget(ez_gfx_hal::QueueKind::Graphics, completions.graphics),
    );
    trim_staging_to_aggregate_budget(context, completions, &mut stale);
    observe_staging_high_water(context);
    for allocation in stale {
        free_native_allocation(&mut context.native, allocation).map_err(map_allocation)?;
    }
    trim_counter_scratch(&mut context.counter_scratch);
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct StagingCompletions {
    transfer: u64,
    graphics: u64,
}

fn staging_completions(context: &mut ContextState) -> Result<StagingCompletions> {
    let transfer = completed_transfer_native(&mut context.native).map_err(map_allocation)?;
    let graphics = completed_native_frame_value(&mut context.native)?;
    Ok(StagingCompletions { transfer, graphics })
}

/// Sums retained staging capacity across every transfer cache.
///
/// Adds backend texture staging to the legacy three-pool aggregate so the
/// shared trim loop terminates on texture-only pressure and counts texture
/// evictions toward its ceiling. Saturating addition keeps the total from
/// wrapping under pathological retention.
pub(super) fn all_transfer_cache_retained(context: &ContextState) -> u64 {
    aggregate_staging_retained(context)
        .saturating_add(retained_native_texture_staging(&context.native))
}

/// Sums retained staging capacity across the shared, stride, and counter pools.
///
/// Saturating addition keeps diagnostics honest: the total must never wrap even
/// under pathological retention.
pub(super) fn aggregate_staging_retained(context: &ContextState) -> u64 {
    let mut total = context.staging.retained_bytes();
    for pool in context.buffer_pool.values() {
        total = total.saturating_add(pool.retained_bytes());
    }
    total.saturating_add(context.counter_pool.retained_bytes())
}

/// Records the actual aggregate peak after a staging-pool mutation or burst.
///
/// The scan only reads pool storage and performs no allocation.
pub(super) fn observe_staging_high_water(context: &mut ContextState) {
    context.staging_high_water_bytes = context
        .staging_high_water_bytes
        .max(aggregate_staging_retained(context));
}

fn trim_staging_to_aggregate_budget_after_graphics(
    context: &mut ContextState,
    graphics: u64,
    stale: &mut Vec<NativeAllocation>,
) -> Result<()> {
    if aggregate_staging_retained(context) <= ez_gfx_hal::DEFAULT_STAGING_AGGREGATE_BUDGET {
        return Ok(());
    }
    let transfer = completed_transfer_native(&mut context.native).map_err(map_allocation)?;
    trim_staging_to_aggregate_budget(context, StagingCompletions { transfer, graphics }, stale);
    Ok(())
}

/// Evicts completed buckets until summed retention fits the aggregate budget.
///
/// Per-pool ceilings cannot bound the stride map, which grows one pool per
/// element width: this pass evicts the largest completed bucket across every
/// pool until the total fits. Eviction order never affects reuse quality
/// because `take` always selects the smallest fitting bucket. Exact ties
/// prefer shared, then counter, then stride pools, so evicted byte totals stay
/// deterministic; in-flight buckets are never candidates and may keep the
/// total over budget until the next call. Pushes evictions into `stale` for
/// the caller to free; allocates nothing itself.
pub(super) fn trim_staging_to_aggregate_budget(
    context: &mut ContextState,
    completed: StagingCompletions,
    stale: &mut Vec<NativeAllocation>,
) {
    // Completion queries fail pre-device admission; without a texture
    // timeline the texture cache simply sits out this pass.
    let texture_completed = completed_texture_transfer_native(&mut context.native).ok();
    trim_staging_to_aggregate_budget_inner(context, completed, texture_completed, stale);
}

/// Infallible eviction core shared by the pressure entry points.
fn trim_staging_to_aggregate_budget_inner(
    context: &mut ContextState,
    completed: StagingCompletions,
    texture_completed: Option<u64>,
    stale: &mut Vec<NativeAllocation>,
) {
    // One uniform eviction order across every cache: the blanket trait impl
    // covers the three context pools while the texture adapter dispatches to
    // the backend pool, so global pressure evicts the largest retained cache
    // first regardless of which manager grew it. Exact ties prefer shared,
    // then counter, then stride, then texture pools, so evicted byte totals
    // stay deterministic; in-flight buckets are never candidates.
    // The ceiling covers every transfer cache, including backend texture
    // staging: texture-only retention must trigger trimming too, and evicting
    // a texture bucket must count toward termination. Legacy diagnostics keep
    // reporting the three context pools via `aggregate_staging_retained`.
    while all_transfer_cache_retained(context) > ez_gfx_hal::DEFAULT_STAGING_AGGREGATE_BUDGET {
        let mut texture_cache = TextureStaging(&mut context.native);
        let mut caches: Vec<EvictionCache<'_>> = Vec::with_capacity(4 + context.buffer_pool.len());
        caches.push(EvictionCache {
            staging: &mut context.staging,
            queue: ez_gfx_hal::QueueKind::Transfer,
            completed: completed.transfer,
        });
        caches.push(EvictionCache {
            staging: &mut context.counter_pool,
            queue: ez_gfx_hal::QueueKind::Graphics,
            completed: completed.graphics,
        });
        for pool in context.buffer_pool.values_mut() {
            caches.push(EvictionCache {
                staging: pool,
                queue: ez_gfx_hal::QueueKind::Graphics,
                completed: completed.graphics,
            });
        }
        if let Some(texture_completed) = texture_completed {
            caches.push(EvictionCache {
                staging: &mut texture_cache,
                queue: ez_gfx_hal::QueueKind::TextureTransfer,
                completed: texture_completed,
            });
        }
        let mut best: Option<usize> = None;
        let mut best_capacity = 0_u64;
        for (index, entry) in caches.iter().enumerate() {
            if let Some(capacity) = entry
                .staging
                .largest_completed_capacity(entry.queue, entry.completed)
                && (best.is_none() || capacity > best_capacity)
            {
                best = Some(index);
                best_capacity = capacity;
            }
        }
        let Some(index) = best else {
            break;
        };
        let entry = &mut caches[index];
        match entry
            .staging
            .pop_largest_completed(entry.queue, entry.completed)
        {
            Some(allocation) => stale.push(allocation),
            // Unreachable: the capacity was just observed above, so a bucket
            // must still be there; break anyway to guard against logic drift.
            None => break,
        }
    }
}

/// Backend texture-staging adapter for the shared eviction order.
///
/// Texture pools live on the transfer timeline only; foreign queues never
/// match, so a misdirected trim observes and evicts nothing.
struct EvictionCache<'a> {
    staging: &'a mut dyn ReclaimableStaging<Staging = NativeAllocation>,
    queue: ez_gfx_hal::QueueKind,
    completed: u64,
}

struct TextureStaging<'a>(&'a mut NativeContext);

impl ReclaimableStaging for TextureStaging<'_> {
    type Staging = NativeAllocation;

    fn retained_bytes(&self) -> u64 {
        retained_native_texture_staging(self.0)
    }

    fn largest_completed_capacity(
        &self,
        queue: ez_gfx_hal::QueueKind,
        completed: u64,
    ) -> Option<u64> {
        (queue == ez_gfx_hal::QueueKind::TextureTransfer)
            .then(|| largest_native_texture_staging(self.0, completed))
            .flatten()
    }

    fn pop_largest_completed(
        &mut self,
        queue: ez_gfx_hal::QueueKind,
        completed: u64,
    ) -> Option<NativeAllocation> {
        (queue == ez_gfx_hal::QueueKind::TextureTransfer)
            .then(|| pop_largest_native_texture_staging(self.0, completed))
            .flatten()
    }
}

/// Releases retained staging caches down to their finite budgets.
///
/// Trims the shared, buffer, and counter staging pools plus excess counter
/// serialization capacity, freeing evicted buckets natively. Completion-gated
/// buckets owned by in-flight GPU work stay retained. Call on memory pressure
/// or after large streaming bursts — never per frame, since the completion
/// query itself costs a native round trip.
///
/// # Errors
///
/// Returns an error for stale handles, wrong-thread calls, or native failures.
pub fn release_staging_memory(context: ContextHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context.identity.check_thread().map_err(map_lifecycle)?;
        trim_staging_caches(context)
    }))
}

fn require_recording(context: &ContextState) -> Result<()> {
    context
        .identity
        .check_thread_and_health()
        .map_err(map_lifecycle)?;
    if context.frame.state() != ez_gfx_runtime::frame::FrameState::Recording {
        return Err(Error::NotReady);
    }
    Ok(())
}

fn validate_writable_transient(
    context: &ContextState,
    handle: PackedHandle,
    kind: ResourceKind,
) -> Result<()> {
    context
        .identity
        .resolve(handle, kind)
        .map_err(map_lifecycle)?;
    let metadata = context
        .transient_buffers
        .get(&handle)
        .ok_or(Error::InvalidContext)?;
    if metadata.usage != TransientUse::Available {
        return Err(Error::NotReady);
    }
    Ok(())
}

fn checked_element_range(
    element_size: u32,
    capacity: u32,
    start_index: u32,
    count: u32,
) -> Result<(u64, u64)> {
    if element_size == 0 {
        return Err(Error::InvalidArgument);
    }
    let end = start_index
        .checked_add(count)
        .ok_or(Error::InvalidArgument)?;
    if end > capacity {
        return Err(Error::InvalidArgument);
    }
    let offset = u64::from(start_index)
        .checked_mul(u64::from(element_size))
        .ok_or(Error::InvalidArgument)?;
    let size = u64::from(count)
        .checked_mul(u64::from(element_size))
        .ok_or(Error::InvalidArgument)?;
    Ok((offset, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_element_ranges_accept_boundaries_and_reject_overflow() {
        assert_eq!(checked_element_range(4, 8, 2, 3), Ok((8, 12)));
        assert_eq!(checked_element_range(4, 8, 8, 0), Ok((32, 0)));
        assert_eq!(
            checked_element_range(4, 8, 7, 2),
            Err(Error::InvalidArgument)
        );
        assert_eq!(
            checked_element_range(u32::MAX, u32::MAX, u32::MAX, 2),
            Err(Error::InvalidArgument)
        );
    }

    #[test]
    fn counter_scratch_trim_bounds_pathological_retention() {
        let mut scratch = vec![0x5a; COUNTER_SCRATCH_RETAIN_LIMIT + 1];
        trim_counter_scratch(&mut scratch);
        assert!(scratch.is_empty());
        assert!(scratch.capacity() <= COUNTER_SCRATCH_RETAIN_LIMIT);
        // Steady-state sizes keep their warm capacity allocation-free.
        scratch.reserve(1024);
        let warm = scratch.capacity();
        trim_counter_scratch(&mut scratch);
        assert_eq!(scratch.capacity(), warm);
    }

    #[test]
    fn counter_payload_places_elements_at_shared_aligned_offset() {
        let command = [0x5a; 20];
        let mut scratch = Vec::new();
        let payload = counter_payload(&mut scratch, &command, 7).unwrap();
        let offset = usize::try_from(COUNTER_BUFFER_ELEMENT_OFFSET).unwrap();

        assert_eq!(&payload[..4], &7_u32.to_le_bytes());
        assert!(payload[4..offset].iter().all(|byte| *byte == 0));
        assert_eq!(&payload[offset..], &command);
        assert_eq!(payload.len(), offset + command.len());
        // A second equal-size write must reuse capacity without reallocating.
        let capacity = scratch.capacity();
        let payload = counter_payload(&mut scratch, &command, 3).unwrap();
        assert_eq!(&payload[..4], &3_u32.to_le_bytes());
        assert_eq!(scratch.capacity(), capacity);
    }
}
