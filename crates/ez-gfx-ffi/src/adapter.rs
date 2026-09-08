use ez_gfx::{AdapterClass, AdapterInfo, AdapterReport, Backend, ContextOptions, raw};

use super::{EzGfxAdapterDesc, EzGfxAdapterInfo, EzGfxResult, catch_status};

// A zero count keeps default ranking and requires a null pointer; a count of
// one selects by stable identity. Anything else fails before any native call.
pub(crate) fn apply_adapter_selection(
    options: ContextOptions,
    count: u32,
    adapter: *const EzGfxAdapterDesc,
) -> Result<ContextOptions, EzGfxResult> {
    match (count, adapter.is_null()) {
        (0, true) => Ok(options),
        (1, false) => {
            // SAFETY: `adapter` is non-null, and the caller keeps readable,
            // properly aligned storage for one `EzGfxAdapterDesc` alive
            // through this read.
            let request = unsafe { adapter.read() };
            if request.allow_software > 1 || request.stable_id == [0; 16] {
                return Err(EzGfxResult::InvalidArgument);
            }
            Ok(options.with_adapter(request.stable_id, request.allow_software == 1))
        }
        _ => Err(EzGfxResult::InvalidArgument),
    }
}

fn backend_code(backend: Backend) -> u8 {
    match backend {
        Backend::Vulkan => 1,
        Backend::Dx12 => 2,
        Backend::Metal => 3,
    }
}

fn adapter_info(info: &AdapterInfo, report: &AdapterReport) -> EzGfxAdapterInfo {
    EzGfxAdapterInfo {
        stable_id: info.stable_id(),
        backend: backend_code(info.backend()),
        adapter_class: match info.class() {
            AdapterClass::Software => 0,
            AdapterClass::Other => 1,
            AdapterClass::Integrated => 2,
            AdapterClass::Discrete => 3,
        },
        admitted: u8::from(report.admitted()),
        software_rejected: u8::from(report.software_rejected()),
        error_count: u32::try_from(report.errors().len()).unwrap_or(u32::MAX),
    }
}

fn saturating_count(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}

#[unsafe(no_mangle)]
/// Returns the number of enumerated adapters without creating anything.
///
/// # Safety
///
/// A non-null `out_count` must address one writable, aligned `u32` for this call.
pub unsafe extern "C" fn ez_gfx_adapter_count(out_count: *mut u32) -> EzGfxResult {
    catch_status(|| {
        if out_count.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `out_count` is non-null, and the caller keeps writable,
        // properly aligned storage for one `u32` alive through this write.
        unsafe { out_count.write(saturating_count(raw::enumerate_adapters().len())) };
        EzGfxResult::Ok
    })
}

#[unsafe(no_mangle)]
/// Enumerates adapters with admission diagnostics under one software policy.
///
/// A zero capacity queries the total with a null buffer; otherwise the call
/// fills at most `capacity` entries and reports how many it wrote. The
/// `admitted`, `software_rejected`, and `error_count` fields diagnose each
/// rejection without creating anything.
///
/// # Safety
///
/// A non-null `out_adapters` must address `capacity` writable, aligned
/// entries, and a non-null `out_written` one writable, aligned `u32`, for
/// this call.
pub unsafe extern "C" fn ez_gfx_adapters_query(
    allow_software: u8,
    out_adapters: *mut EzGfxAdapterInfo,
    capacity: u32,
    out_written: *mut u32,
) -> EzGfxResult {
    catch_status(|| {
        if allow_software > 1 || out_written.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        if capacity == 0 {
            if !out_adapters.is_null() {
                return EzGfxResult::InvalidArgument;
            }
            // SAFETY: `out_written` is non-null with live aligned `u32` storage.
            unsafe {
                out_written.write(saturating_count(raw::enumerate_adapters().len()));
            }
            return EzGfxResult::Ok;
        }
        if out_adapters.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let reports = raw::query_adapter_report(allow_software == 1);
        let writable = usize::try_from(capacity)
            .unwrap_or(usize::MAX)
            .min(reports.len());
        // SAFETY: `capacity` is nonzero and bounds the caller range;
        // `writable` never exceeds it, so exactly this prefix is writable.
        let out = unsafe { core::slice::from_raw_parts_mut(out_adapters, writable) };
        for (slot, report) in out.iter_mut().zip(&reports) {
            *slot = adapter_info(report.adapter(), report);
        }
        // SAFETY: `out_written` is non-null with live aligned `u32` storage.
        unsafe { out_written.write(saturating_count(writable)) };
        EzGfxResult::Ok
    })
}
