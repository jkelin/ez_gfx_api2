fn rollback_transient_internment(context: &mut ContextState) {
    for buffer in context.transient_buffers.values_mut() {
        if buffer.usage == super::TransientUse::Interned(context.frame_serial) {
            buffer.usage = super::TransientUse::Available;
        }
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
