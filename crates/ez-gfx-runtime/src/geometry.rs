use std::collections::{BTreeMap, HashMap};

use ez_gfx_core::handle::PackedHandle;
use ez_gfx_hal::{CompletionToken, DEFAULT_STAGING_POLICY, QueueKind, staging_bucket_size};

const MAX_HEAP_NAME_BYTES: usize = 255;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Describes a live contiguous allocation within a geometry heap.
pub struct GeometryUpload {
    /// Generation- and owner-bearing public allocation identity.
    pub handle: PackedHandle,
    /// Zero-based element position where the allocation begins.
    pub first_element: u32,
    /// Number of elements in the allocation.
    pub element_count: u32,
    /// Byte position where the allocation begins.
    pub byte_offset: u64,
    /// Number of bytes reserved for the upload.
    pub byte_size: u64,
}

#[derive(Clone, Debug)]
struct LiveRange {
    heap: HeapIdentity,
    upload: GeometryUpload,
    ready: Option<CompletionToken>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum HeapIdentity {
    Vertex(String),
    Index,
}

#[derive(Clone, Debug)]
struct Heap {
    capacity: u64,
    stride: u64,
    free: BTreeMap<u64, u64>,
    ready: Option<CompletionToken>,
}

impl Heap {
    fn new(capacity: u64, stride: u64) -> Self {
        Self {
            capacity,
            stride,
            free: BTreeMap::from([(0, capacity)]),
            ready: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Tracks named vertex and global index heap ranges and upload readiness.
pub struct GeometryManager {
    vertex: BTreeMap<String, Heap>,
    index: Option<Heap>,
    live: HashMap<PackedHandle, LiveRange>,
}

impl GeometryManager {
    /// Creates an empty geometry manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates one validated named vertex heap.
    ///
    /// # Errors
    ///
    /// Rejects invalid names, capacities, strides, and duplicate names.
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
        self.vertex
            .insert(name.to_owned(), Heap::new(capacity, stride));
        Ok(())
    }

    /// Creates the single packed-u32 index heap.
    ///
    /// # Errors
    ///
    /// Rejects capacities smaller than one index and duplicate heaps.
    pub fn create_index_heap(&mut self, capacity: u64) -> Result<(), GeometryError> {
        if capacity < 4 {
            return Err(GeometryError::InvalidCapacity);
        }
        if self.index.is_some() {
            return Err(GeometryError::DuplicateHeap);
        }
        self.index = Some(Heap::new(capacity, 4));
        Ok(())
    }

    /// Allocates a contiguous vertex range and associates its public identity.
    ///
    /// # Errors
    ///
    /// Rejects unknown heaps, duplicate handles, zero counts, stride mismatch, and exhaustion.
    pub fn reserve_vertices(
        &mut self,
        name: &str,
        count: u32,
        element_size: u64,
        handle: PackedHandle,
    ) -> Result<GeometryUpload, GeometryError> {
        if self.live.contains_key(&handle) {
            return Err(GeometryError::DuplicateAllocation);
        }
        let heap = self
            .vertex
            .get_mut(name)
            .ok_or(GeometryError::UnknownHeap)?;
        if element_size != heap.stride {
            return Err(GeometryError::StrideMismatch);
        }
        let upload = reserve(heap, count, handle)?;
        self.live.insert(
            handle,
            LiveRange {
                heap: HeapIdentity::Vertex(name.to_owned()),
                upload,
                ready: None,
            },
        );
        Ok(upload)
    }

    /// Allocates a contiguous index range and associates its public identity.
    ///
    /// # Errors
    ///
    /// Rejects a missing heap, duplicate handle, zero count, or exhaustion.
    pub fn reserve_indices(
        &mut self,
        count: u32,
        handle: PackedHandle,
    ) -> Result<GeometryUpload, GeometryError> {
        if self.live.contains_key(&handle) {
            return Err(GeometryError::DuplicateAllocation);
        }
        let heap = self.index.as_mut().ok_or(GeometryError::UnknownHeap)?;
        let upload = reserve(heap, count, handle)?;
        self.live.insert(
            handle,
            LiveRange {
                heap: HeapIdentity::Index,
                upload,
                ready: None,
            },
        );
        Ok(upload)
    }

    /// Returns one live allocation's range.
    ///
    /// # Errors
    ///
    /// Returns `UnknownAllocation` for stale, freed, or foreign identities.
    pub fn allocation(&self, handle: PackedHandle) -> Result<GeometryUpload, GeometryError> {
        self.live
            .get(&handle)
            .map(|range| range.upload)
            .ok_or(GeometryError::UnknownAllocation)
    }

    /// Records the upload token for one allocation.
    ///
    /// # Errors
    ///
    /// Rejects stale identities and timeline regression.
    pub fn mark_ready(
        &mut self,
        handle: PackedHandle,
        token: CompletionToken,
    ) -> Result<(), GeometryError> {
        let range = self
            .live
            .get_mut(&handle)
            .ok_or(GeometryError::UnknownAllocation)?;
        if range
            .ready
            .is_some_and(|current| current.queue == token.queue && current.value >= token.value)
        {
            return Err(GeometryError::TimelineRegression);
        }
        range.ready = Some(token);
        let heap = match &range.heap {
            HeapIdentity::Vertex(name) => self.vertex.get_mut(name),
            HeapIdentity::Index => self.index.as_mut(),
        }
        .ok_or(GeometryError::UnknownHeap)?;
        if heap.ready.is_none_or(|current| current.value < token.value) {
            heap.ready = Some(token);
        }
        Ok(())
    }

    /// Releases a live vertex allocation after the caller establishes GPU safety.
    ///
    /// # Errors
    ///
    /// Rejects stale identities and handles belonging to another heap.
    pub fn free_vertices(
        &mut self,
        name: &str,
        handle: PackedHandle,
    ) -> Result<GeometryUpload, GeometryError> {
        let range = self
            .live
            .get(&handle)
            .cloned()
            .ok_or(GeometryError::UnknownAllocation)?;
        if range.heap != HeapIdentity::Vertex(name.to_owned()) {
            return Err(GeometryError::WrongHeap);
        }
        self.live.remove(&handle);
        let heap = self
            .vertex
            .get_mut(name)
            .ok_or(GeometryError::UnknownHeap)?;
        release_range(heap, range.upload)?;
        Ok(range.upload)
    }

    /// Releases a live vertex allocation using its recorded heap ownership.
    ///
    /// # Errors
    ///
    /// Rejects stale identities and non-vertex allocation handles.
    pub fn free_vertex(&mut self, handle: PackedHandle) -> Result<GeometryUpload, GeometryError> {
        let range = self
            .live
            .get(&handle)
            .cloned()
            .ok_or(GeometryError::UnknownAllocation)?;
        let HeapIdentity::Vertex(name) = &range.heap else {
            return Err(GeometryError::WrongHeap);
        };
        self.live.remove(&handle);
        let heap = self
            .vertex
            .get_mut(name)
            .ok_or(GeometryError::UnknownHeap)?;
        release_range(heap, range.upload)?;
        Ok(range.upload)
    }

    /// Releases a live index allocation after the caller establishes GPU safety.
    ///
    /// # Errors
    ///
    /// Rejects stale identities and vertex allocation handles.
    pub fn free_indices(&mut self, handle: PackedHandle) -> Result<GeometryUpload, GeometryError> {
        let range = self
            .live
            .get(&handle)
            .cloned()
            .ok_or(GeometryError::UnknownAllocation)?;
        if range.heap != HeapIdentity::Index {
            return Err(GeometryError::WrongHeap);
        }
        self.live.remove(&handle);
        let heap = self.index.as_mut().ok_or(GeometryError::UnknownHeap)?;
        release_range(heap, range.upload)?;
        Ok(range.upload)
    }

    /// Returns the latest token for a named heap.
    pub fn vertex_ready(&self, name: &str) -> Option<CompletionToken> {
        self.vertex.get(name).and_then(|heap| heap.ready)
    }

    /// Returns the latest token for the index heap.
    pub fn index_ready(&self) -> Option<CompletionToken> {
        self.index.as_ref().and_then(|heap| heap.ready)
    }

    /// Extends a named vertex heap without moving existing logical ranges.
    ///
    /// # Errors
    ///
    /// Rejects unknown heaps, nonincreasing capacities, or capacities whose
    /// element offsets cannot be represented by the public `u32` range.
    pub fn grow_vertex_heap(&mut self, name: &str, new_capacity: u64) -> Result<(), GeometryError> {
        let heap = self
            .vertex
            .get_mut(name)
            .ok_or(GeometryError::UnknownHeap)?;
        grow_heap(heap, new_capacity)
    }

    /// Extends the packed-u32 index heap without moving existing logical ranges.
    ///
    /// # Errors
    ///
    /// Rejects a missing heap, nonincreasing capacities, or capacities whose
    /// index offsets cannot be represented by the public `u32` range.
    pub fn grow_index_heap(&mut self, new_capacity: u64) -> Result<(), GeometryError> {
        let heap = self.index.as_mut().ok_or(GeometryError::UnknownHeap)?;
        grow_heap(heap, new_capacity)
    }

    /// Removes an empty vertex heap.
    ///
    /// # Errors
    ///
    /// Rejects unknown or nonempty heaps.
    pub fn remove_vertex_heap(&mut self, name: &str) -> Result<(), GeometryError> {
        if self
            .live
            .values()
            .any(|range| range.heap == HeapIdentity::Vertex(name.to_owned()))
        {
            return Err(GeometryError::HeapNotEmpty);
        }
        self.vertex
            .remove(name)
            .map(|_| ())
            .ok_or(GeometryError::UnknownHeap)
    }

    /// Removes an empty index heap.
    ///
    /// # Errors
    ///
    /// Rejects a missing or nonempty heap.
    pub fn remove_index_heap(&mut self) -> Result<(), GeometryError> {
        if self
            .live
            .values()
            .any(|range| range.heap == HeapIdentity::Index)
        {
            return Err(GeometryError::HeapNotEmpty);
        }
        self.index
            .take()
            .map(|_| ())
            .ok_or(GeometryError::UnknownHeap)
    }
}

fn grow_heap(heap: &mut Heap, new_capacity: u64) -> Result<(), GeometryError> {
    if new_capacity <= heap.capacity || new_capacity / heap.stride > u64::from(u32::MAX) {
        return Err(GeometryError::InvalidCapacity);
    }

    let old_capacity = heap.capacity;
    let extension = new_capacity - old_capacity;
    heap.capacity = new_capacity;
    if let Some((&previous, &previous_size)) = heap.free.range(..old_capacity).next_back()
        && previous.checked_add(previous_size) == Some(old_capacity)
    {
        heap.free.remove(&previous);
        let size = previous_size
            .checked_add(extension)
            .ok_or(GeometryError::CapacityExceeded)?;
        heap.free.insert(previous, size);
    } else {
        heap.free.insert(old_capacity, extension);
    }
    Ok(())
}

fn reserve(
    heap: &mut Heap,
    count: u32,
    handle: PackedHandle,
) -> Result<GeometryUpload, GeometryError> {
    if count == 0 {
        return Err(GeometryError::InvalidCount);
    }
    let byte_size = u64::from(count)
        .checked_mul(heap.stride)
        .ok_or(GeometryError::CapacityExceeded)?;
    let (&offset, &available) = heap
        .free
        .iter()
        .find(|(_, available)| **available >= byte_size)
        .ok_or(GeometryError::CapacityExceeded)?;
    heap.free.remove(&offset);
    if available > byte_size {
        heap.free.insert(offset + byte_size, available - byte_size);
    }
    let first_element =
        u32::try_from(offset / heap.stride).map_err(|_| GeometryError::CapacityExceeded)?;
    Ok(GeometryUpload {
        handle,
        first_element,
        element_count: count,
        byte_offset: offset,
        byte_size,
    })
}

fn release_range(heap: &mut Heap, upload: GeometryUpload) -> Result<(), GeometryError> {
    let mut start = upload.byte_offset;
    let mut size = upload.byte_size;
    if let Some((&previous, &previous_size)) = heap.free.range(..start).next_back()
        && previous.checked_add(previous_size) == Some(start)
    {
        heap.free.remove(&previous);
        start = previous;
        size = size
            .checked_add(previous_size)
            .ok_or(GeometryError::CapacityExceeded)?;
    }
    if let Some((&next, &next_size)) = heap.free.range(start..).next()
        && start.checked_add(size) == Some(next)
    {
        heap.free.remove(&next);
        size = size
            .checked_add(next_size)
            .ok_or(GeometryError::CapacityExceeded)?;
    }
    let end = start
        .checked_add(size)
        .ok_or(GeometryError::CapacityExceeded)?;
    if end > heap.capacity || heap.free.insert(start, size).is_some() {
        return Err(GeometryError::InvalidFree);
    }
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

#[derive(Clone, Debug, Default)]
/// Reuses staging buffers without an artificial slot limit.
pub struct StagingPool {
    entries: Vec<StagingEntry>,
}

impl StagingPool {
    /// Creates an empty staging pool.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// A retired slot is reusable only after its transfer value completes and only if it fits.
    ///
    /// # Errors
    ///
    /// Returns `InvalidCount` if size is zero or `StagingPoolExhausted` only if
    /// the slot index cannot be represented.
    ///
    /// # Panics
    ///
    /// Existing slot indices were validated when inserted.
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
    /// A public allocation handle is already live.
    DuplicateAllocation,
    /// The requested vertex or index heap does not exist.
    UnknownHeap,
    /// The requested allocation is stale or unknown.
    UnknownAllocation,
    /// The allocation belongs to another heap kind or name.
    WrongHeap,
    /// A heap still owns live ranges.
    HeapNotEmpty,
    /// The requested vertex element size differs from the heap stride.
    StrideMismatch,
    /// A reservation exceeds heap capacity or supported offset arithmetic.
    CapacityExceeded,
    /// A free range overlaps or exceeds the heap.
    InvalidFree,
    /// A readiness token does not advance its queue timeline.
    TimelineRegression,
    /// The staging slot index cannot be represented.
    StagingPoolExhausted,
    /// The staging slot index is unknown or the slot is not checked out.
    InvalidStagingSlot,
    /// A staging retirement token did not belong to the transfer queue.
    WrongQueue,
}
