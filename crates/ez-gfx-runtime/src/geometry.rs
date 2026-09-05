use std::collections::BTreeMap;

use ez_gfx_hal::{CompletionToken, DEFAULT_STAGING_POLICY, QueueKind, staging_bucket_size};

const MAX_HEAP_NAME_BYTES: usize = 255;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Describes a contiguous reservation within a geometry heap.
pub struct GeometryUpload {
    /// Zero-based element position where the reservation begins.
    pub first_element: u32,
    /// Byte position where the reservation begins.
    pub byte_offset: u64,
    /// Number of bytes reserved for the upload.
    pub byte_size: u64,
}

#[derive(Clone, Debug)]
struct Heap {
    capacity: u64,
    stride: u64,
    used: u64,
    ready: Option<CompletionToken>,
    latest_reservation: Option<GeometryUpload>,
}

#[derive(Clone, Debug, Default)]
/// Tracks vertex and index heap allocation and upload readiness.
pub struct GeometryManager {
    /// Named vertex heaps with independent capacities and strides.
    vertex: BTreeMap<String, Heap>,
    /// Optional heap containing packed 32-bit indices.
    index: Option<Heap>,
}

impl GeometryManager {
    /// Creates an empty geometry manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Names are bounded ASCII identifiers; zero capacity/stride and duplicate names are rejected.
    ///
    /// # Errors
    ///
    /// Returns `InvalidName` for an invalid heap name, `InvalidCapacity` for zero capacity, `InvalidStride` for a zero or oversized stride, or `DuplicateHeap` when the name already exists.
    pub fn create_vertex_heap(
        &mut self,
        name: &str,
        capacity: u64,
        stride: u64,
    ) -> Result<(), GeometryError> {
        validate_name(name)?;
        if capacity == 0 {
            return Err(GeometryError::InvalidCapacity);
        }
        if stride == 0 || stride > u64::from(u32::MAX) {
            return Err(GeometryError::InvalidStride);
        }
        if self.vertex.contains_key(name) {
            return Err(GeometryError::DuplicateHeap);
        }
        self.vertex.insert(
            name.to_owned(),
            Heap {
                capacity,
                stride,
                used: 0,
                ready: None,
                latest_reservation: None,
            },
        );
        Ok(())
    }

    /// Only one u32 index heap exists; capacity must hold at least one complete index.
    ///
    /// # Errors
    ///
    /// Returns `InvalidCapacity` when capacity is less than four bytes or `DuplicateHeap` when an index heap already exists.
    pub fn create_index_heap(&mut self, capacity: u64) -> Result<(), GeometryError> {
        if capacity < 4 {
            return Err(GeometryError::InvalidCapacity);
        }
        if self.index.is_some() {
            return Err(GeometryError::DuplicateHeap);
        }
        self.index = Some(Heap {
            capacity,
            stride: 4,
            used: 0,
            ready: None,
            latest_reservation: None,
        });
        Ok(())
    }

    /// Reserves contiguous space for vertices in the named heap.
    ///
    /// # Errors
    ///
    /// Returns an error if the heap is unknown, the count is zero, the element size differs from the heap stride, or the reservation exceeds supported capacity or arithmetic.
    pub fn reserve_vertices(
        &mut self,
        name: &str,
        count: u32,
        element_size: u64,
    ) -> Result<GeometryUpload, GeometryError> {
        let heap = self
            .vertex
            .get_mut(name)
            .ok_or(GeometryError::UnknownHeap)?;
        if count == 0 {
            return Err(GeometryError::InvalidCount);
        }
        if element_size != heap.stride {
            return Err(GeometryError::StrideMismatch);
        }
        reserve(heap, count)
    }

    /// Reserves contiguous space for 32-bit indices.
    ///
    /// # Errors
    ///
    /// Returns an error if the index heap is unknown, the count is zero, or the reservation exceeds supported capacity or arithmetic.
    pub fn reserve_indices(&mut self, count: u32) -> Result<GeometryUpload, GeometryError> {
        if count == 0 {
            return Err(GeometryError::InvalidCount);
        }
        reserve(
            self.index.as_mut().ok_or(GeometryError::UnknownHeap)?,
            count,
        )
    }

    /// Only the immediately latest reservation is rollbackable. A newer reservation permanently
    /// supersedes the previous candidate, even when the newer reservation is rolled back.
    ///
    /// # Errors
    ///
    /// Returns `UnknownHeap` if the vertex heap does not exist or `InvalidRollback` if the upload is not the current rollback candidate.
    pub fn rollback_vertices(
        &mut self,
        name: &str,
        upload: GeometryUpload,
    ) -> Result<(), GeometryError> {
        rollback(
            self.vertex
                .get_mut(name)
                .ok_or(GeometryError::UnknownHeap)?,
            upload,
        )
    }

    /// Reclaims the immediately latest index reservation.
    ///
    /// A successful rollback clears the candidate rather than restoring an older reservation.
    ///
    /// # Errors
    ///
    /// Returns `UnknownHeap` if the index heap does not exist or `InvalidRollback` if the upload is not the current rollback candidate.
    pub fn rollback_indices(&mut self, upload: GeometryUpload) -> Result<(), GeometryError> {
        rollback(
            self.index.as_mut().ok_or(GeometryError::UnknownHeap)?,
            upload,
        )
    }

    /// Records the completion token that makes a vertex heap usable.
    ///
    /// # Errors
    ///
    /// Returns `UnknownHeap` if the vertex heap does not exist or `TimelineRegression` if the token does not advance the same queue timeline.
    pub fn mark_vertex_ready(
        &mut self,
        name: &str,
        token: CompletionToken,
    ) -> Result<(), GeometryError> {
        mark_ready(
            self.vertex
                .get_mut(name)
                .ok_or(GeometryError::UnknownHeap)?,
            token,
        )
    }

    /// Records the completion token that makes the index heap usable.
    ///
    /// # Errors
    ///
    /// Returns `UnknownHeap` if the index heap does not exist or `TimelineRegression` if the token does not advance the same queue timeline.
    pub fn mark_index_ready(&mut self, token: CompletionToken) -> Result<(), GeometryError> {
        mark_ready(
            self.index.as_mut().ok_or(GeometryError::UnknownHeap)?,
            token,
        )
    }

    /// Returns the completion token recorded for the named vertex heap.
    pub fn vertex_ready(&self, name: &str) -> Option<CompletionToken> {
        self.vertex.get(name).and_then(|heap| heap.ready)
    }
    /// Returns the completion token recorded for the index heap.
    pub fn index_ready(&self) -> Option<CompletionToken> {
        self.index.as_ref().and_then(|heap| heap.ready)
    }
    /// Removes the named vertex heap and its allocation state.
    ///
    /// # Errors
    ///
    /// Returns `UnknownHeap` if the named vertex heap does not exist.
    pub fn remove_vertex_heap(&mut self, name: &str) -> Result<(), GeometryError> {
        self.vertex
            .remove(name)
            .map(|_| ())
            .ok_or(GeometryError::UnknownHeap)
    }
    /// Removes the index heap and its allocation state.
    ///
    /// # Errors
    ///
    /// Returns `UnknownHeap` if no index heap exists.
    pub fn remove_index_heap(&mut self) -> Result<(), GeometryError> {
        self.index
            .take()
            .map(|_| ())
            .ok_or(GeometryError::UnknownHeap)
    }
}

/// Appends a contiguous reservation to a heap.
///
/// # Errors
///
/// Returns `CapacityExceeded` if size arithmetic overflows, the reservation exceeds heap capacity, or its first element cannot fit in `u32`.
fn reserve(heap: &mut Heap, count: u32) -> Result<GeometryUpload, GeometryError> {
    let byte_size = u64::from(count)
        .checked_mul(heap.stride)
        .ok_or(GeometryError::CapacityExceeded)?;
    let end = heap
        .used
        .checked_add(byte_size)
        .ok_or(GeometryError::CapacityExceeded)?;
    if end > heap.capacity {
        return Err(GeometryError::CapacityExceeded);
    }
    let first_element =
        u32::try_from(heap.used / heap.stride).map_err(|_| GeometryError::CapacityExceeded)?;
    let upload = GeometryUpload {
        first_element,
        byte_offset: heap.used,
        byte_size,
    };
    heap.latest_reservation = Some(upload);
    heap.used = end;
    Ok(upload)
}

/// Reclaims only the immediately latest reservation.
///
/// # Errors
///
/// Returns `InvalidRollback` unless the upload exactly matches the current rollback candidate.
fn rollback(heap: &mut Heap, upload: GeometryUpload) -> Result<(), GeometryError> {
    // Public fields are forgeable, and rollback must not revive an older superseded candidate.
    if heap.latest_reservation != Some(upload) {
        return Err(GeometryError::InvalidRollback);
    }
    heap.latest_reservation.take();
    heap.used = upload.byte_offset;
    Ok(())
}

/// Updates a heap's readiness token without regressing the same queue timeline.
///
/// # Errors
///
/// Returns `TimelineRegression` if the token's value does not exceed the current value for the same queue.
fn mark_ready(heap: &mut Heap, token: CompletionToken) -> Result<(), GeometryError> {
    if heap
        .ready
        .is_some_and(|current| current.queue == token.queue && current.value >= token.value)
    {
        return Err(GeometryError::TimelineRegression);
    }
    heap.ready = Some(token);
    Ok(())
}

/// Accepts a bounded, nonempty ASCII heap name using letters, digits, `_`, `-`, or `.`.
///
/// # Errors
///
/// Returns `InvalidName` if the name is empty, exceeds 255 bytes, or contains characters other than ASCII letters, digits, `_`, `-`, or `.`.
fn validate_name(name: &str) -> Result<(), GeometryError> {
    if name.is_empty()
        || name.len() > MAX_HEAP_NAME_BYTES
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(GeometryError::InvalidName);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Identifies a staging allocation by its pool index.
pub struct StagingSlot(u32);

#[derive(Clone, Copy, Debug)]
struct StagingEntry {
    capacity: u64,
    in_use: bool,
    retirement: Option<CompletionToken>,
}

#[derive(Clone, Debug)]
/// Reuses and allocates staging buffers up to a fixed slot limit.
pub struct StagingPool {
    /// Maximum number of staging slots that may exist.
    capacity: u32,
    /// Allocated staging slots and their reuse state.
    entries: Vec<StagingEntry>,
}

impl StagingPool {
    /// Creates an empty staging pool with the given slot limit.
    ///
    /// # Errors
    ///
    /// Returns `InvalidCapacity` if the slot limit is zero.
    pub fn new(capacity: u32) -> Result<Self, GeometryError> {
        if capacity == 0 {
            return Err(GeometryError::InvalidCapacity);
        }
        Ok(Self {
            capacity,
            entries: Vec::new(),
        })
    }

    /// A retired slot is reusable only after its transfer value completes and only if it fits.
    ///
    /// # Errors
    ///
    /// Returns `InvalidCount` if size is zero or `StagingPoolExhausted` if no reusable slot is available at the slot limit.
    ///
    /// # Panics
    ///
    /// The staging pool capacity bounds every slot index conversion.
    pub fn checkout(
        &mut self,
        size: u64,
        completed_transfer: u64,
    ) -> Result<StagingSlot, GeometryError> {
        let bucket = staging_bucket_size(size, DEFAULT_STAGING_POLICY)
            .map_err(|_| GeometryError::InvalidCount)?;
        if let Some((index, entry)) = self.entries.iter_mut().enumerate().find(|(_, entry)| {
            !entry.in_use
                && entry.capacity >= size
                && entry
                    .retirement
                    .is_none_or(|token| token.value <= completed_transfer)
        }) {
            entry.in_use = true;
            entry.retirement = None;
            return Ok(StagingSlot(
                u32::try_from(index).expect("validated index fits u32"),
            ));
        }
        if self.entries.len() >= self.capacity as usize {
            return Err(GeometryError::StagingPoolExhausted);
        }
        self.entries.push(StagingEntry {
            capacity: bucket,
            in_use: true,
            retirement: None,
        });
        let slot = u32::try_from(self.entries.len() - 1)
            .map_err(|_| GeometryError::StagingPoolExhausted)?;
        Ok(StagingSlot(slot))
    }

    /// Returns the allocated byte capacity of a staging slot.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError::InvalidStagingSlot`] when the slot does not exist.
    pub fn slot_capacity(&self, slot: StagingSlot) -> Result<u64, GeometryError> {
        self.entries
            .get(slot.0 as usize)
            .map(|entry| entry.capacity)
            .ok_or(GeometryError::InvalidStagingSlot)
    }

    /// Releases a submitted slot after associating its transfer completion token.
    ///
    /// # Errors
    ///
    /// Returns `InvalidStagingSlot` if the slot does not exist or is not currently checked out,
    /// or `WrongQueue` if the token is not from the transfer timeline consumed by checkout.
    pub fn retire(
        &mut self,
        slot: StagingSlot,
        token: CompletionToken,
    ) -> Result<(), GeometryError> {
        let entry = self
            .entries
            .get_mut(slot.0 as usize)
            .ok_or(GeometryError::InvalidStagingSlot)?;
        if !entry.in_use {
            return Err(GeometryError::InvalidStagingSlot);
        }
        // `checkout` receives only the transfer timeline value, so other queues cannot gate reuse.
        if token.queue != QueueKind::Transfer {
            return Err(GeometryError::WrongQueue);
        }
        entry.in_use = false;
        entry.retirement = Some(token);
        Ok(())
    }

    /// Releases a slot immediately when no transfer was submitted.
    ///
    /// # Errors
    ///
    /// Returns `InvalidStagingSlot` if the slot does not exist or is not currently checked out.
    pub fn release_unsubmitted(&mut self, slot: StagingSlot) -> Result<(), GeometryError> {
        let entry = self
            .entries
            .get_mut(slot.0 as usize)
            .ok_or(GeometryError::InvalidStagingSlot)?;
        if !entry.in_use {
            return Err(GeometryError::InvalidStagingSlot);
        }
        entry.in_use = false;
        entry.retirement = None;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Reports geometry heap, reservation, readiness, and staging pool failures.
pub enum GeometryError {
    /// A heap name is empty, too long, or contains unsupported characters.
    InvalidName,
    /// A heap or staging pool capacity is invalid.
    InvalidCapacity,
    /// A vertex stride is zero or cannot be represented by supported element indexing.
    InvalidStride,
    /// A reservation or staging checkout requested zero elements or bytes.
    InvalidCount,
    /// A heap already exists for the requested name or index storage.
    DuplicateHeap,
    /// The requested vertex or index heap does not exist.
    UnknownHeap,
    /// The requested vertex element size differs from the heap stride.
    StrideMismatch,
    /// A reservation exceeds heap capacity or supported offset arithmetic.
    CapacityExceeded,
    /// A readiness token does not advance its queue timeline.
    TimelineRegression,
    /// No reusable staging slot exists and the slot limit has been reached.
    StagingPoolExhausted,
    /// The staging slot index is unknown or the slot is not checked out.
    InvalidStagingSlot,
    /// A staging retirement token did not belong to the transfer queue.
    WrongQueue,
    /// The reservation is not the heap's most recent allocation.
    InvalidRollback,
}
