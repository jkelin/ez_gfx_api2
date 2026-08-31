#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ObservationCounts {
    pub runtime_events: u32,
    pub diagnostics: u32,
    pub dropped: u64,
}

pub fn drain_bounded<E>(
    limit: u32,
    mut event: impl FnMut() -> Result<(bool, u64), E>,
    mut diagnostic: impl FnMut() -> Result<(bool, u64), E>,
) -> Result<ObservationCounts, E> {
    let mut counts = ObservationCounts::default();
    for _ in 0..limit {
        let (present, dropped) = event()?;
        counts.dropped = counts.dropped.saturating_add(dropped);
        if !present {
            break;
        }
        counts.runtime_events += 1;
    }
    for _ in 0..limit {
        let (present, dropped) = diagnostic()?;
        counts.dropped = counts.dropped.saturating_add(dropped);
        if !present {
            break;
        }
        counts.diagnostics += 1;
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_stops_on_empty_and_saturates_drops() {
        let mut events = [(true, u64::MAX), (false, 1)].into_iter();
        let counts =
            drain_bounded(4, || Ok::<_, ()>(events.next().unwrap()), || Ok((false, 2))).unwrap();
        assert_eq!(
            counts,
            ObservationCounts {
                runtime_events: 1,
                diagnostics: 0,
                dropped: u64::MAX
            }
        );
    }

    #[test]
    fn drain_honors_each_queue_limit() {
        let counts = drain_bounded(2, || Ok::<_, ()>((true, 0)), || Ok((true, 0))).unwrap();
        assert_eq!((counts.runtime_events, counts.diagnostics), (2, 2));
    }
}
