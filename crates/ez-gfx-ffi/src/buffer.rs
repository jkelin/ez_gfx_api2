use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
    thread::{self, ThreadId},
};

use ez_gfx::{
    DrawIndexedCommand,
    raw::{self, ContextHandle, ResourceIdentity},
};

use super::{EZ_GFX_MAX_BOUNDARY_BYTES, EzGfxFrame, EzGfxResult};

const TAG: u64 = 0xB7 << 56;
const TAG_MASK: u64 = 0xFF << 56;
const KIND_SHIFT: u32 = 48;
const GENERATION_SHIFT: u32 = 24;
const FIELD_MASK: u64 = (1 << 24) - 1;
const FIELD_MASK_U32: u32 = (1 << 24) - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kind {
    Structured = 1,
    Counted = 2,
}

struct Entry {
    owner: ContextHandle,
    kind: Kind,
    generation: u32,
    element_size: u32,
    element_count: u32,
    bytes: Vec<u8>,
    published_count: u32,
    creator: ThreadId,
}

struct Slot {
    generation: u32,
    entry: Option<Entry>,
    retired: bool,
}
#[derive(Default)]
struct Registry {
    slots: Vec<Slot>,
    free: Vec<u32>,
}
static BUFFERS: LazyLock<Mutex<Registry>> = LazyLock::new(|| Mutex::new(Registry::default()));
type MaterializedKey = (EzGfxFrame, u64);
type MaterializedResources = HashMap<MaterializedKey, ResourceIdentity>;
static MATERIALIZED: LazyLock<Mutex<MaterializedResources>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn encode(index: u32, generation: u32, kind: Kind) -> u64 {
    TAG | ((kind as u64) << KIND_SHIFT)
        | (u64::from(generation) << GENERATION_SHIFT)
        | u64::from(index + 1)
}

fn decode(handle: u64) -> Result<(u32, u32, Kind), EzGfxResult> {
    if handle & TAG_MASK != TAG {
        return Err(EzGfxResult::InvalidContext);
    }
    let kind = match (handle >> KIND_SHIFT) & 0xFF {
        1 => Kind::Structured,
        2 => Kind::Counted,
        _ => return Err(EzGfxResult::InvalidContext),
    };
    let slot = handle & FIELD_MASK;
    let generation = (handle >> GENERATION_SHIFT) & FIELD_MASK;
    if slot == 0 || generation == 0 {
        return Err(EzGfxResult::InvalidContext);
    }
    Ok((
        u32::try_from(slot - 1).map_err(|_| EzGfxResult::InvalidContext)?,
        u32::try_from(generation).map_err(|_| EzGfxResult::InvalidContext)?,
        kind,
    ))
}

pub(crate) fn insert(
    owner: ContextHandle,
    kind: Kind,
    element_size: u32,
    element_count: u32,
) -> Result<u64, EzGfxResult> {
    let byte_count = (element_size as usize)
        .checked_mul(element_count as usize)
        .filter(|size| {
            element_size != 0 && element_count != 0 && *size <= EZ_GFX_MAX_BOUNDARY_BYTES
        })
        .ok_or(EzGfxResult::InvalidArgument)?;
    let mut registry = BUFFERS.lock().map_err(|_| EzGfxResult::NativeFailure)?;
    while let Some(index) = registry.free.pop() {
        let slot = &mut registry.slots[index as usize];
        if slot.retired || slot.entry.is_some() {
            continue;
        }
        let generation = slot.generation;
        slot.entry = Some(Entry {
            owner,
            kind,
            generation,
            element_size,
            element_count,
            bytes: vec![0; byte_count],
            published_count: 0,
            creator: thread::current().id(),
        });
        return Ok(encode(index, generation, kind));
    }
    let index = u32::try_from(registry.slots.len()).map_err(|_| EzGfxResult::NativeFailure)?;
    if index >= FIELD_MASK_U32 {
        return Err(EzGfxResult::NativeFailure);
    }
    registry.slots.push(Slot {
        generation: 1,
        entry: Some(Entry {
            owner,
            kind,
            generation: 1,
            element_size,
            element_count,
            bytes: vec![0; byte_count],
            published_count: 0,
            creator: thread::current().id(),
        }),
        retired: false,
    });
    Ok(encode(index, 1, kind))
}

fn with_entry<T>(
    handle: u64,
    owner: ContextHandle,
    kind: Kind,
    operation: impl FnOnce(&mut Entry) -> Result<T, EzGfxResult>,
) -> Result<T, EzGfxResult> {
    let (index, generation, encoded_kind) = decode(handle)?;
    let mut registry = BUFFERS.lock().map_err(|_| EzGfxResult::NativeFailure)?;
    let slot = registry
        .slots
        .get_mut(index as usize)
        .ok_or(EzGfxResult::InvalidContext)?;
    let entry = slot.entry.as_mut().ok_or(EzGfxResult::InvalidContext)?;
    if slot.retired
        || slot.generation != generation
        || entry.generation != generation
        || encoded_kind != kind
        || entry.kind != kind
        || entry.owner != owner
        || entry.creator != thread::current().id()
    {
        return Err(EzGfxResult::InvalidContext);
    }
    operation(entry)
}
fn require_not_materialized(handle: u64) -> Result<(), EzGfxResult> {
    // Creator-thread validation serializes imports, but the registry still rejects live snapshots.
    let materialized = MATERIALIZED
        .lock()
        .map_err(|_| EzGfxResult::NativeFailure)?;
    if materialized
        .keys()
        .any(|(_, candidate)| *candidate == handle)
    {
        return Err(EzGfxResult::NotReady);
    }
    Ok(())
}

pub(crate) fn write(
    handle: u64,
    owner: ContextHandle,
    kind: Kind,
    start_index: u32,
    element_size: u32,
    bytes: &[u8],
) -> Result<(), EzGfxResult> {
    require_not_materialized(handle)?;
    with_entry(handle, owner, kind, |entry| {
        if entry.element_size != element_size || bytes.len() % element_size as usize != 0 {
            return Err(EzGfxResult::InvalidArgument);
        }
        let count = bytes.len() / element_size as usize;
        let start = start_index as usize;
        let end = start
            .checked_add(count)
            .filter(|end| *end <= entry.element_count as usize)
            .ok_or(EzGfxResult::InvalidArgument)?;
        let byte_start = start
            .checked_mul(element_size as usize)
            .ok_or(EzGfxResult::InvalidArgument)?;
        let byte_end = end
            .checked_mul(element_size as usize)
            .ok_or(EzGfxResult::InvalidArgument)?;
        entry.bytes[byte_start..byte_end].copy_from_slice(bytes);
        if kind == Kind::Counted {
            let published = u32::try_from(end).map_err(|_| EzGfxResult::InvalidArgument)?;
            entry.published_count = entry.published_count.max(published);
        }
        Ok(())
    })
}

pub(crate) fn publish(handle: u64, owner: ContextHandle, count: u32) -> Result<(), EzGfxResult> {
    require_not_materialized(handle)?;
    with_entry(handle, owner, Kind::Counted, |entry| {
        if count > entry.element_count {
            return Err(EzGfxResult::InvalidArgument);
        }
        entry.published_count = count;
        Ok(())
    })
}

pub(crate) fn materialize(
    frame: EzGfxFrame,
    handle: u64,
    kind: Kind,
) -> Result<ResourceIdentity, EzGfxResult> {
    if let Some(resource) = MATERIALIZED
        .lock()
        .map_err(|_| EzGfxResult::NativeFailure)?
        .get(&(frame, handle))
        .cloned()
    {
        return Ok(resource);
    }
    let frame_entry = crate::frame::get(frame)?;
    let resource = with_entry(handle, frame_entry.owner, kind, |entry| match kind {
        Kind::Structured => {
            let raw_handle = raw::acquire_structured_raw(
                frame_entry.owner,
                entry.element_size,
                entry.element_count,
            )
            .map_err(EzGfxResult::from)?;
            if let Err(error) = raw::write_structured_raw(
                frame_entry.owner,
                raw_handle,
                0,
                entry.element_count,
                entry.element_size,
                &entry.bytes,
            ) {
                raw::release_structured(frame_entry.owner, raw_handle);
                return Err(error.into());
            }
            Ok(ResourceIdentity::Structured(raw_handle))
        }
        Kind::Counted => {
            if entry.element_size as usize != core::mem::size_of::<DrawIndexedCommand>() {
                return Err(EzGfxResult::InvalidArgument);
            }
            let raw_handle = raw::acquire_indirect(frame_entry.owner, entry.element_count)
                .map_err(EzGfxResult::from)?;
            let commands = entry
                .bytes
                .chunks_exact(entry.element_size as usize)
                .map(bytemuck::pod_read_unaligned)
                .collect::<Vec<DrawIndexedCommand>>();
            let result =
                raw::write_indirect(frame_entry.owner, raw_handle, 0, &commands).and_then(|()| {
                    raw::publish_compute_indirect_count(
                        frame_entry.owner,
                        raw_handle,
                        entry.published_count,
                    )
                });
            if let Err(error) = result {
                raw::release_indirect(frame_entry.owner, raw_handle);
                return Err(error.into());
            }
            Ok(ResourceIdentity::Indirect(raw_handle))
        }
    })?;
    MATERIALIZED
        .lock()
        .map_err(|_| EzGfxResult::NativeFailure)?
        .insert((frame, handle), resource.clone());
    Ok(resource)
}

pub(crate) fn remove(handle: u64, owner: ContextHandle, kind: Kind) -> Result<(), EzGfxResult> {
    let (index, generation, encoded_kind) = decode(handle)?;
    let mut registry = BUFFERS.lock().map_err(|_| EzGfxResult::NativeFailure)?;
    let slot = registry
        .slots
        .get_mut(index as usize)
        .ok_or(EzGfxResult::InvalidContext)?;
    let entry = slot.entry.as_ref().ok_or(EzGfxResult::InvalidContext)?;
    if slot.generation != generation
        || entry.generation != generation
        || encoded_kind != kind
        || entry.kind != kind
        || entry.owner != owner
        || entry.creator != thread::current().id()
    {
        return Err(EzGfxResult::InvalidContext);
    }
    slot.entry = None;
    if slot.generation == FIELD_MASK_U32 {
        slot.retired = true;
    } else {
        slot.generation += 1;
        registry.free.push(index);
    }
    drop(registry);
    MATERIALIZED
        .lock()
        .map_err(|_| EzGfxResult::NativeFailure)?
        .retain(|(_, candidate), _| *candidate != handle);
    Ok(())
}

pub(crate) fn clear_frame(frame: EzGfxFrame) {
    // Per-frame native materializations expire; context-owned CPU contents remain reusable.
    if let Ok(mut materialized) = MATERIALIZED.lock() {
        materialized.retain(|(candidate, _), _| *candidate != frame);
    }
}

pub(crate) fn remove_owner(owner: ContextHandle) {
    // Context teardown retires every descendant identity, including never-imported buffers.
    let mut owned_handles = Vec::new();
    if let Ok(mut registry) = BUFFERS.lock() {
        for (index, slot) in registry.slots.iter_mut().enumerate() {
            if let Some(entry) = slot.entry.as_ref()
                && entry.owner == owner
            {
                if let Ok(index) = u32::try_from(index) {
                    owned_handles.push(encode(index, entry.generation, entry.kind));
                }
                slot.entry = None;
                slot.retired = true;
            }
        }
    }
    if let Ok(mut materialized) = MATERIALIZED.lock() {
        materialized.retain(|(_, handle), _| !owned_handles.contains(handle));
    }
}
