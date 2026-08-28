use ez_gfx_core::Backend;
use ez_gfx_runtime::observability::{
    DiagnosticLevel, Observability, RuntimePhase, RuntimeRecord, RuntimeStatus,
};

fn record(id: u64) -> RuntimeRecord {
    RuntimeRecord {
        correlation_id: id,
        resource: 7,
        backend: Backend::Dx12,
        phase: RuntimePhase::Submit,
        status: RuntimeStatus::Ok,
    }
}

#[test]
fn bounded_event_queue_is_fifo_and_reports_drops_once() {
    let mut stream = Observability::new(2, 1).unwrap();
    stream.push_event(record(1));
    stream.push_event(record(2));
    stream.push_event(record(3));
    assert_eq!(stream.poll_event(), (Some(record(1)), 1));
    assert_eq!(stream.poll_event(), (Some(record(2)), 0));
    assert_eq!(stream.poll_event(), (None, 0));
}

#[test]
fn diagnostics_are_bounded_and_independent_from_events() {
    let mut stream = Observability::new(1, 1).unwrap();
    stream.push_diagnostic(DiagnosticLevel::Error, record(4));
    stream.push_diagnostic(DiagnosticLevel::Warning, record(5));
    assert_eq!(
        stream.poll_diagnostic(),
        (Some((DiagnosticLevel::Error, record(4))), 1)
    );
    assert_eq!(stream.poll_event(), (None, 0));
    assert!(Observability::new(0, 1).is_err());
    assert!(Observability::new(1, 0).is_err());
}
