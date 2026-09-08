use std::{
    sync::{LazyLock, Mutex},
    thread::{self, ThreadId},
};

use ez_gfx::{raw, raw::ContextHandle};

use super::{EzGfxFrame, EzGfxResult};

const FRAME_TAG: u64 = 0xE7 << 56;
const FRAME_TAG_MASK: u64 = 0xFF << 56;
const KIND_SHIFT: u32 = 48;
const KIND_MASK: u64 = 0xFF << KIND_SHIFT;
const GENERATION_SHIFT: u32 = 24;
const FIELD_MASK: u64 = (1 << 24) - 1;
const FIELD_MASK_U32: u32 = (1 << 24) - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FrameKind {
    Surface = 1,
    RenderTarget = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FrameState {
    Recording,
    Ended,
    Aborted,
    ContextDestroyed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrameEntry {
    pub(crate) owner: ContextHandle,
    pub(crate) kind: FrameKind,
    generation: u32,
    state: FrameState,
    serial: u64,
    creator: ThreadId,
}

struct Slot {
    generation: u32,
    entry: Option<FrameEntry>,
    retired: bool,
}

#[derive(Default)]
struct Registry {
    slots: Vec<Slot>,
    free: Vec<u32>,
}

static FRAMES: LazyLock<Mutex<Registry>> = LazyLock::new(|| Mutex::new(Registry::default()));

fn encode(slot: u32, generation: u32, kind: FrameKind) -> EzGfxFrame {
    FRAME_TAG
        | ((kind as u64) << KIND_SHIFT)
        | (u64::from(generation) << GENERATION_SHIFT)
        | (u64::from(slot) + 1)
}

fn decode(frame: EzGfxFrame) -> Result<(u32, u32, FrameKind), EzGfxResult> {
    if frame & FRAME_TAG_MASK != FRAME_TAG {
        return Err(EzGfxResult::InvalidContext);
    }
    let kind = match (frame & KIND_MASK) >> KIND_SHIFT {
        1 => FrameKind::Surface,
        2 => FrameKind::RenderTarget,
        _ => return Err(EzGfxResult::InvalidContext),
    };
    let encoded_slot = frame & FIELD_MASK;
    let generation = (frame >> GENERATION_SHIFT) & FIELD_MASK;
    if encoded_slot == 0 || generation == 0 {
        return Err(EzGfxResult::InvalidContext);
    }
    Ok((
        u32::try_from(encoded_slot - 1).map_err(|_| EzGfxResult::InvalidContext)?,
        u32::try_from(generation).map_err(|_| EzGfxResult::InvalidContext)?,
        kind,
    ))
}

fn lookup(frame: EzGfxFrame) -> Result<FrameEntry, EzGfxResult> {
    let (index, generation, kind) = decode(frame)?;
    let registry = FRAMES.lock().map_err(|_| EzGfxResult::NativeFailure)?;
    let slot = registry
        .slots
        .get(index as usize)
        .ok_or(EzGfxResult::InvalidContext)?;
    let entry = slot.entry.ok_or(EzGfxResult::InvalidContext)?;
    if slot.retired
        || slot.generation != generation
        || entry.generation != generation
        || entry.kind != kind
        || entry.state != FrameState::Recording
        || entry.creator != thread::current().id()
    {
        return Err(EzGfxResult::InvalidContext);
    }
    Ok(entry)
}

fn validate_serial(entry: FrameEntry, serial: u64) -> Result<(), EzGfxResult> {
    if serial == entry.serial {
        Ok(())
    } else {
        Err(EzGfxResult::InvalidContext)
    }
}

fn current_serial(entry: FrameEntry) -> Result<(), EzGfxResult> {
    let serial = raw::current_frame_serial(entry.owner).map_err(EzGfxResult::from)?;
    validate_serial(entry, serial)
}

pub(crate) fn insert(
    owner: ContextHandle,
    kind: FrameKind,
    serial: u64,
) -> Result<EzGfxFrame, EzGfxResult> {
    if serial == 0 {
        return Err(EzGfxResult::InvalidContext);
    }
    let mut registry = FRAMES.lock().map_err(|_| EzGfxResult::NativeFailure)?;
    while let Some(index) = registry.free.pop() {
        let slot = &mut registry.slots[index as usize];
        if slot.retired
            || slot
                .entry
                .is_some_and(|entry| entry.state == FrameState::Recording)
        {
            continue;
        }
        let generation = slot.generation;
        slot.entry = Some(FrameEntry {
            owner,
            kind,
            generation,
            state: FrameState::Recording,
            serial,
            creator: thread::current().id(),
        });
        return Ok(encode(index, generation, kind));
    }

    // The 24-bit slot field cannot represent another frame; the caller aborts the raw frame.
    let index = u32::try_from(registry.slots.len()).map_err(|_| EzGfxResult::NativeFailure)?;
    if index >= FIELD_MASK_U32 {
        return Err(EzGfxResult::NativeFailure);
    }
    registry.slots.push(Slot {
        generation: 1,
        entry: Some(FrameEntry {
            owner,
            kind,
            generation: 1,
            state: FrameState::Recording,
            serial,
            creator: thread::current().id(),
        }),
        retired: false,
    });
    Ok(encode(index, 1, kind))
}

pub(crate) fn get(frame: EzGfxFrame) -> Result<FrameEntry, EzGfxResult> {
    let entry = lookup(frame)?;
    current_serial(entry)?;
    Ok(entry)
}

fn transition(frame: EzGfxFrame, state: FrameState) -> Result<FrameEntry, EzGfxResult> {
    let (index, generation, kind) = decode(frame)?;
    let mut registry = FRAMES.lock().map_err(|_| EzGfxResult::NativeFailure)?;
    let slot = registry
        .slots
        .get_mut(index as usize)
        .ok_or(EzGfxResult::InvalidContext)?;
    let mut entry = slot.entry.ok_or(EzGfxResult::InvalidContext)?;
    if slot.retired
        || slot.generation != generation
        || entry.generation != generation
        || entry.kind != kind
        || entry.state != FrameState::Recording
        || entry.creator != thread::current().id()
    {
        return Err(EzGfxResult::InvalidContext);
    }
    entry.state = state;
    slot.entry = Some(entry);
    if slot.generation == FIELD_MASK_U32 {
        // Generation exhaustion retires the slot so a stale wire value never becomes live again.
        slot.retired = true;
    } else {
        slot.generation += 1;
        registry.free.push(index);
    }
    Ok(entry)
}

pub(crate) fn remove(frame: EzGfxFrame, state: FrameState) -> Result<FrameEntry, EzGfxResult> {
    let entry = lookup(frame)?;
    let serial = current_serial(entry);
    // A terminal attempt consumes the handle even when raw liveness or serial validation fails.
    let entry = transition(frame, state)?;
    serial.map(|()| entry)
}

pub(crate) fn remove_owner_frame(owner: ContextHandle) -> Option<FrameEntry> {
    let frame = {
        let registry = FRAMES.lock().ok()?;
        registry
            .slots
            .iter()
            .enumerate()
            .find_map(|(index, slot)| {
                let entry = slot.entry?;
                (entry.owner == owner
                    && entry.state == FrameState::Recording
                    && entry.creator == thread::current().id())
                .then(|| {
                    encode(
                        u32::try_from(index).expect("frame slot fits its wire field"),
                        entry.generation,
                        entry.kind,
                    )
                })
            })?
    };
    let entry = lookup(frame).ok()?;
    // Context teardown invalidates the registry entry even if raw frame state has already drifted.
    let _ = current_serial(entry);
    transition(frame, FrameState::ContextDestroyed).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(raw: u64) -> ContextHandle {
        ContextHandle::from_raw(raw).unwrap()
    }

    #[test]
    fn stale_wrong_kind_and_foreign_thread_frames_are_rejected() {
        let owner = context((1 << 20) | 101);
        let frame = insert(owner, FrameKind::Surface, 7).unwrap();
        let entry = lookup(frame).unwrap();
        assert_eq!(entry.serial, 7);
        assert_eq!(validate_serial(entry, 8), Err(EzGfxResult::InvalidContext));
        assert_eq!(
            lookup(frame ^ (3 << KIND_SHIFT)),
            Err(EzGfxResult::InvalidContext)
        );
        std::thread::spawn(move || {
            assert_eq!(lookup(frame), Err(EzGfxResult::InvalidContext));
        })
        .join()
        .unwrap();
        assert_eq!(lookup(frame).unwrap().serial, 7);
        assert_eq!(
            transition(frame, FrameState::Aborted).unwrap().state,
            FrameState::Aborted
        );
        assert_eq!(lookup(frame), Err(EzGfxResult::InvalidContext));
        let replacement = insert(owner, FrameKind::Surface, 8).unwrap();
        assert_ne!(replacement, frame);
        let replacement_entry = lookup(replacement).unwrap();
        assert_eq!(replacement_entry.serial, 8);
        assert_eq!(replacement_entry.generation, decode(replacement).unwrap().1);
        transition(replacement, FrameState::Ended).unwrap();
    }

    #[test]
    fn owner_removal_records_terminal_state_and_preserves_other_owners() {
        let first = context((1 << 20) | 201);
        let second = context((1 << 20) | 202);
        let first_frame = insert(first, FrameKind::Surface, 11).unwrap();
        let second_frame = insert(second, FrameKind::RenderTarget, 13).unwrap();

        assert_eq!(
            remove_owner_frame(first).unwrap().state,
            FrameState::ContextDestroyed
        );
        assert!(remove_owner_frame(first).is_none());
        assert_eq!(lookup(first_frame), Err(EzGfxResult::InvalidContext));
        assert_eq!(lookup(second_frame).unwrap().owner, second);
        transition(second_frame, FrameState::Aborted).unwrap();
    }
}
