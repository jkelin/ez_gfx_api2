use std::collections::BTreeMap;

use ez_gfx_hal::CompletionToken;

const MAX_HEAP_NAME_BYTES: usize = 255;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryUpload {
    pub first_element: u32,
    pub byte_offset: u64,
    pub byte_size: u64,
}

#[derive(Clone, Debug)]
struct Heap {
    capacity: u64,
    stride: u64,
    used: u64,
    ready: Option<CompletionToken>,
}

#[derive(Clone, Debug, Default)]
pub struct GeometryManager {
    vertex: BTreeMap<String, Heap>,
    index: Option<Heap>,
}

impl GeometryManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Names are bounded ASCII identifiers; zero capacity/stride and duplicate names are rejected.
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
            },
        );
        Ok(())
    }

    /// Only one u32 index heap exists; capacity must hold at least one complete index.
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
        });
        Ok(())
    }

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

    pub fn reserve_indices(&mut self, count: u32) -> Result<GeometryUpload, GeometryError> {
        if count == 0 {
            return Err(GeometryError::InvalidCount);
        }
        reserve(
            self.index.as_mut().ok_or(GeometryError::UnknownHeap)?,
            count,
        )
    }

    /// Rollback is accepted only for the most recent reservation, preventing overlapping reuse.
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

    pub fn rollback_indices(&mut self, upload: GeometryUpload) -> Result<(), GeometryError> {
        rollback(
            self.index.as_mut().ok_or(GeometryError::UnknownHeap)?,
            upload,
        )
    }

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

    pub fn mark_index_ready(&mut self, token: CompletionToken) -> Result<(), GeometryError> {
        mark_ready(
            self.index.as_mut().ok_or(GeometryError::UnknownHeap)?,
            token,
        )
    }

    pub fn vertex_ready(&self, name: &str) -> Option<CompletionToken> {
        self.vertex.get(name).and_then(|heap| heap.ready)
    }
    pub fn index_ready(&self) -> Option<CompletionToken> {
        self.index.as_ref().and_then(|heap| heap.ready)
    }
    pub fn remove_vertex_heap(&mut self, name: &str) -> Result<(), GeometryError> {
        self.vertex
            .remove(name)
            .map(|_| ())
            .ok_or(GeometryError::UnknownHeap)
    }
    pub fn remove_index_heap(&mut self) -> Result<(), GeometryError> {
        self.index
            .take()
            .map(|_| ())
            .ok_or(GeometryError::UnknownHeap)
    }
}

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
    heap.used = end;
    Ok(upload)
}

fn rollback(heap: &mut Heap, upload: GeometryUpload) -> Result<(), GeometryError> {
    if upload.byte_offset.checked_add(upload.byte_size) != Some(heap.used) {
        return Err(GeometryError::InvalidRollback);
    }
    heap.used = upload.byte_offset;
    Ok(())
}

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
pub struct StagingSlot(u32);

#[derive(Clone, Copy, Debug)]
struct StagingEntry {
    capacity: u64,
    in_use: bool,
    retirement: Option<CompletionToken>,
}

#[derive(Clone, Debug)]
pub struct StagingPool {
    capacity: u32,
    entries: Vec<StagingEntry>,
}

impl StagingPool {
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
    pub fn checkout(
        &mut self,
        size: u64,
        completed_transfer: u64,
    ) -> Result<StagingSlot, GeometryError> {
        if size == 0 {
            return Err(GeometryError::InvalidCount);
        }
        if let Some((index, entry)) = self.entries.iter_mut().enumerate().find(|(_, entry)| {
            !entry.in_use
                && entry.capacity >= size
                && entry
                    .retirement
                    .is_none_or(|token| token.value <= completed_transfer)
        }) {
            entry.in_use = true;
            entry.retirement = None;
            return Ok(StagingSlot(index as u32));
        }
        if self.entries.len() >= self.capacity as usize {
            return Err(GeometryError::StagingPoolExhausted);
        }
        self.entries.push(StagingEntry {
            capacity: size,
            in_use: true,
            retirement: None,
        });
        Ok(StagingSlot((self.entries.len() - 1) as u32))
    }

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
        entry.in_use = false;
        entry.retirement = Some(token);
        Ok(())
    }

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
pub enum GeometryError {
    InvalidName,
    InvalidCapacity,
    InvalidStride,
    InvalidCount,
    DuplicateHeap,
    UnknownHeap,
    StrideMismatch,
    CapacityExceeded,
    TimelineRegression,
    StagingPoolExhausted,
    InvalidStagingSlot,
    InvalidRollback,
}
