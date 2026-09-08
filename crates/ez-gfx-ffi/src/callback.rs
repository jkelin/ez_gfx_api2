//! Owner-thread context event callbacks replacing host polling.
//!
//! Registration replaces any previous callback; a null callback clears the
//! slot. Delivery happens only inside explicit C entry points on the calling
//! (owner) thread — never on worker threads — by draining the same raw
//! queues the removed poll exports exposed. A reentrant dispatch attempt
//! fails, and a panicking callback is unregistered exactly once.

use std::{
    cell::Cell,
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use core::ffi::c_void;

use ez_gfx::raw::ContextHandle;

use super::{
    EzGfxEvent, EzGfxEventCallback, EzGfxEventKind, EzGfxResult, EzGfxRuntimeRecord,
    EzGfxUploadEvent,
};
use ez_gfx::{UploadResource, UploadStatus};

/// Maximum events delivered by one dispatch, matching safe dispatch bounds.
const DISPATCH_BOUND: usize = 4096;

thread_local! {
    static INVOKING_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

/// Reports whether the current thread is inside an application callback.
pub(crate) fn is_invoking() -> bool {
    INVOKING_CALLBACK.get()
}

struct Slot {
    callback: unsafe extern "C" fn(*const EzGfxEvent, *mut c_void),
    user_data: usize,
    dispatching: bool,
}

/// Auxiliary per-frame C tracking for presentation and ordered readback requests.
/// The raw frame owns lifecycle; this map only annotates live frames and is
/// swept on frame consumption and teardown.
pub(crate) struct FrameAux {
    pub(crate) owner: u64,
    pub(crate) surface: u64,
    pub(crate) target: u64,
    pub(crate) readback_sources: Vec<ReadbackSource>,
}

static CALLBACKS: LazyLock<Mutex<HashMap<u64, Slot>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static FRAME_AUX: LazyLock<Mutex<HashMap<u64, FrameAux>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_READBACK_REQUEST: AtomicU64 = AtomicU64::new(1);

fn blank_event(kind: EzGfxEventKind) -> EzGfxEvent {
    EzGfxEvent {
        kind,
        _pad_kind: [0; 7],
        upload: EzGfxUploadEvent {
            resource: 0,
            resource_kind: 0,
            status: 0,
            error: 0,
            _padding: [0; 5],
        },
        record: EzGfxRuntimeRecord {
            correlation_id: 0,
            resource: 0,
            backend: 0,
            phase: 0,
            status: 0,
            _padding: [0; 5],
        },
        level: 0,
        _pad_level: [0; 7],
        dropped: 0,
        readback_request_id: 0,
        readback_texture: 0,
        readback_width: 0,
        readback_height: 0,
        readback_byte_count: 0,
        readback_bytes: core::ptr::null(),
    }
}

fn map_upload(resource: UploadResource, status: UploadStatus) -> (u64, u8, u8, u8) {
    let (resource, resource_kind) = match resource {
        UploadResource::Texture(handle) => (handle.into_raw(), 1),
        UploadResource::Vertex(handle) => (handle.into_raw(), 2),
        UploadResource::Index(handle) => (handle.into_raw(), 3),
    };
    let (status, error) = match status {
        UploadStatus::SourceStaged => (1, 0),
        UploadStatus::DeviceReady => (2, 0),
        UploadStatus::Failed(error) => (3, error as u8),
        UploadStatus::Cancelled => (4, 0),
    };
    (resource, resource_kind, status, error)
}

/// Registers (or with a null callback, clears) the context event callback,
/// then delivers already-pending events to the new registration.
pub(crate) fn register(
    context: ContextHandle,
    callback: EzGfxEventCallback,
    user_data: *mut c_void,
) -> EzGfxResult {
    let key = context.into_raw();
    let Some(callback) = callback else {
        if let Ok(mut slots) = CALLBACKS.lock() {
            slots.remove(&key);
            return EzGfxResult::Ok;
        }
        return EzGfxResult::NativeFailure;
    };
    match CALLBACKS.lock() {
        Ok(mut slots) => {
            if slots.get(&key).is_some_and(|slot| slot.dispatching) {
                return EzGfxResult::InvalidArgument;
            }
            slots.insert(
                key,
                Slot {
                    callback,
                    user_data: user_data as usize,
                    dispatching: false,
                },
            );
        }
        Err(_) => return EzGfxResult::NativeFailure,
    }
    dispatch(context)
}

/// Drops the callback slot without delivery; context teardown owns this path.
pub(crate) fn remove(context: ContextHandle) {
    let key = context.into_raw();
    if let Ok(mut slots) = CALLBACKS.lock() {
        slots.remove(&key);
    }
    if let Ok(mut aux) = FRAME_AUX.lock() {
        aux.retain(|_, entry| entry.owner != key);
    }
}

/// Rejects dispatch-triggering calls made from inside a callback.
pub(crate) fn check_entry(context: ContextHandle) -> Result<(), EzGfxResult> {
    match CALLBACKS.lock() {
        Ok(slots) => {
            if slots
                .get(&context.into_raw())
                .is_some_and(|slot| slot.dispatching)
            {
                return Err(EzGfxResult::InvalidArgument);
            }
            Ok(())
        }
        Err(_) => Err(EzGfxResult::NativeFailure),
    }
}

/// Records present surface/target raw handles for a live frame.
pub(crate) fn note_frame(frame: u64, owner: u64, surface: u64, target: u64) {
    if let Ok(mut aux) = FRAME_AUX.lock() {
        aux.insert(
            frame,
            FrameAux {
                owner,
                surface,
                target,
                readback_sources: Vec::new(),
            },
        );
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ReadbackSource {
    pub(crate) request_id: u64,
    pub(crate) texture: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Reserves and records one stable correlator in graph insertion order.
pub(crate) fn note_readback_source(
    frame: u64,
    texture: u64,
    width: u32,
    height: u32,
) -> Result<u64, EzGfxResult> {
    let request_id = NEXT_READBACK_REQUEST
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| EzGfxResult::QueueFull)?;
    let mut aux = FRAME_AUX.lock().map_err(|_| EzGfxResult::NativeFailure)?;
    let entry = aux.get_mut(&frame).ok_or(EzGfxResult::InvalidContext)?;
    entry.readback_sources.push(ReadbackSource {
        request_id,
        texture,
        width,
        height,
    });
    Ok(request_id)
}

/// Removes a request whose graph insertion failed after correlator reservation.
pub(crate) fn cancel_readback(frame: u64, request_id: u64) {
    if let Ok(mut aux) = FRAME_AUX.lock()
        && let Some(entry) = aux.get_mut(&frame)
        && entry
            .readback_sources
            .last()
            .is_some_and(|source| source.request_id == request_id)
    {
        entry.readback_sources.pop();
    }
}

/// Consumes auxiliary frame tracking; every terminal frame path owns this call.
pub(crate) fn take_frame(frame: u64) -> Option<FrameAux> {
    FRAME_AUX.lock().ok()?.remove(&frame)
}

pub(crate) struct ReadbackDelivery {
    pub(crate) kind: EzGfxEventKind,
    pub(crate) request_id: u64,
    pub(crate) texture: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bytes: Vec<u8>,
}

struct ResetGuard(u64);

impl Drop for ResetGuard {
    fn drop(&mut self) {
        if let Ok(mut slots) = CALLBACKS.lock() {
            if let Some(slot) = slots.get_mut(&self.0) {
                slot.dispatching = false;
            }
        }
    }
}

struct InvocationGuard;

impl Drop for InvocationGuard {
    fn drop(&mut self) {
        INVOKING_CALLBACK.set(false);
    }
}

type CallbackFn = unsafe extern "C" fn(*const EzGfxEvent, *mut c_void);

fn begin_invoke(key: u64) -> Result<(CallbackFn, usize), EzGfxResult> {
    match CALLBACKS.lock() {
        Ok(mut slots) => {
            let Some(slot) = slots.get_mut(&key) else {
                return Err(EzGfxResult::Ok);
            };
            if slot.dispatching {
                return Err(EzGfxResult::InvalidArgument);
            }
            slot.dispatching = true;
            Ok((slot.callback, slot.user_data))
        }
        Err(_) => Err(EzGfxResult::NativeFailure),
    }
}

fn invoke(callback: CallbackFn, user_data: usize, event: &EzGfxEvent) -> Result<(), EzGfxResult> {
    if INVOKING_CALLBACK.replace(true) {
        return Err(EzGfxResult::InvalidArgument);
    }
    let _guard = InvocationGuard;
    // SAFETY: `event` outlives the call and `user_data` is the registration
    // value; delivery runs on the calling owner thread, never on a worker.
    catch_unwind(AssertUnwindSafe(|| unsafe {
        callback(std::ptr::from_ref(event), user_data as *mut _);
    }))
    .map_err(|_| EzGfxResult::NativeFailure)
}

/// Removes a panicking callback exactly once, mirroring safe unregistration.
fn fail(key: u64) {
    if let Ok(mut slots) = CALLBACKS.lock() {
        slots.remove(&key);
    }
}

/// Delivers one readback or persistent snapshot; bytes are borrowed for the callback only.
pub(crate) fn dispatch_readback(
    context: ContextHandle,
    delivery: &ReadbackDelivery,
) -> EzGfxResult {
    let key = context.into_raw();
    let (callback, user_data) = match begin_invoke(key) {
        Ok(pair) => pair,
        Err(EzGfxResult::Ok) => return EzGfxResult::Ok,
        Err(status) => return status,
    };
    let _guard = ResetGuard(key);
    let mut event = blank_event(delivery.kind);
    event.readback_request_id = delivery.request_id;
    event.readback_texture = delivery.texture;
    event.readback_width = delivery.width;
    event.readback_height = delivery.height;
    event.readback_byte_count = delivery.bytes.len();
    event.readback_bytes = delivery.bytes.as_ptr();
    match invoke(callback, user_data, &event) {
        Ok(()) => EzGfxResult::Ok,
        Err(status) => {
            fail(key);
            status
        }
    }
}
/// Drains queued upload, runtime, and diagnostic transitions into the callback.
pub(crate) fn dispatch(context: ContextHandle) -> EzGfxResult {
    fn deliver(
        key: u64,
        callback: CallbackFn,
        user_data: usize,
        event: &EzGfxEvent,
    ) -> EzGfxResult {
        match invoke(callback, user_data, event) {
            Ok(()) => EzGfxResult::Ok,
            Err(status) => {
                fail(key);
                status
            }
        }
    }
    let key = context.into_raw();
    let (callback, user_data) = match begin_invoke(key) {
        Ok(pair) => pair,
        Err(EzGfxResult::Ok) => return EzGfxResult::Ok,
        Err(status) => return status,
    };
    // The guard holds `dispatching` across every event so a reentrant callback
    // cannot interleave a nested dispatch; it resets on every exit path below.
    let _guard = ResetGuard(key);
    for _ in 0..DISPATCH_BOUND {
        let upload = match ez_gfx::raw::poll_upload_event(context) {
            Ok(event) => event,
            Err(error) => return error.into(),
        };
        let (runtime, runtime_dropped) = match ez_gfx::raw::poll_runtime_event(context) {
            Ok(value) => value,
            Err(error) => return error.into(),
        };
        let (diagnostic, diagnostic_dropped) = match ez_gfx::raw::poll_diagnostic(context) {
            Ok(value) => value,
            Err(error) => return error.into(),
        };
        let dropped = runtime_dropped.saturating_add(diagnostic_dropped);
        let mut pending = false;
        if let Some(event) = upload {
            pending = true;
            let (resource, resource_kind, status, error) = map_upload(event.resource, event.status);
            let mut record = blank_event(EzGfxEventKind::Upload);
            record.upload.resource = resource;
            record.upload.resource_kind = resource_kind;
            record.upload.status = status;
            record.upload.error = error;
            if deliver(key, callback, user_data, &record) != EzGfxResult::Ok {
                return EzGfxResult::NativeFailure;
            }
        }
        if let Some(record) = runtime {
            pending = true;
            let mut event = blank_event(EzGfxEventKind::Runtime);
            event.record.correlation_id = record.correlation_id;
            event.record.resource = record.resource;
            event.record.backend = record.backend as u8;
            event.record.phase = record.phase as u8;
            event.record.status = record.status as u8;
            if deliver(key, callback, user_data, &event) != EzGfxResult::Ok {
                return EzGfxResult::NativeFailure;
            }
        }
        if let Some((level, record)) = diagnostic {
            pending = true;
            let mut event = blank_event(EzGfxEventKind::Diagnostic);
            event.record.correlation_id = record.correlation_id;
            event.record.backend = record.backend as u8;
            event.record.phase = record.phase as u8;
            event.record.status = record.status as u8;
            event.record.resource = record.resource;
            event.level = level as u8;
            if deliver(key, callback, user_data, &event) != EzGfxResult::Ok {
                return EzGfxResult::NativeFailure;
            }
        }
        if dropped != 0 {
            pending = true;
            let mut event = blank_event(EzGfxEventKind::ObservationsDropped);
            event.dropped = dropped;
            if deliver(key, callback, user_data, &event) != EzGfxResult::Ok {
                return EzGfxResult::NativeFailure;
            }
        }
        if !pending {
            break;
        }
    }
    EzGfxResult::Ok
}
