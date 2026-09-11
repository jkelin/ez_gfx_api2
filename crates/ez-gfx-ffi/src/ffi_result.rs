trait IntoFfiResult {
    fn into_ffi_result(self) -> EzGfxResult;
}

impl IntoFfiResult for EzGfxResult {
    fn into_ffi_result(self) -> EzGfxResult {
        self
    }
}

impl IntoFfiResult for ez_gfx::Result<()> {
    fn into_ffi_result(self) -> EzGfxResult {
        self.map_or_else(Into::into, |()| EzGfxResult::Ok)
    }
}

fn catch_status<T: IntoFfiResult>(operation: impl FnOnce() -> T) -> EzGfxResult {
    // Reentrant calls would alias the owner-thread runtime while it dispatches.
    if callback::is_invoking() {
        return EzGfxResult::InvalidArgument;
    }
    catch_unwind(AssertUnwindSafe(operation))
        .map(IntoFfiResult::into_ffi_result)
        .unwrap_or(EzGfxResult::NativeFailure)
}

fn catch_context_destroy<T: IntoFfiResult>(operation: impl FnOnce() -> T) -> EzGfxResult {
    // Any panic or reentrant rejection leaves context teardown unproven.
    if callback::is_invoking() {
        return EzGfxResult::TeardownAbandoned;
    }
    catch_unwind(AssertUnwindSafe(operation))
        .map(IntoFfiResult::into_ffi_result)
        .unwrap_or(EzGfxResult::TeardownAbandoned)
}

#[cfg(test)]
mod context_destroy_tests {
    use super::*;

    #[test]
    fn panic_maps_to_teardown_abandoned() {
        assert_eq!(
            catch_context_destroy(|| -> EzGfxResult { panic!("injected teardown panic") }),
            EzGfxResult::TeardownAbandoned
        );
    }
}

fn catch_frame_terminal<T: IntoFfiResult>(
    frame_handle: EzGfxFrame,
    operation: impl FnOnce() -> T,
) -> EzGfxResult {
    // A panic must still retire the opaque handle and unwind the raw transaction.
    let result = if let Ok(result) = catch_unwind(AssertUnwindSafe(operation)) {
        result.into_ffi_result()
    } else {
        if let Ok(entry) = frame::remove(frame_handle, frame::FrameState::Aborted) {
            callback::take_frame(frame_handle);
            let _ = raw::frame_abort(entry.owner);
        }
        EzGfxResult::NativeFailure
    };
    clear_binding_draft(frame_handle);
    buffer::clear_frame(frame_handle);
    result
}
