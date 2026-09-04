//! Bounded texture asset metadata, mip residency, and worker events.
#![forbid(unsafe_code)]

use parking_lot::Mutex;
use rayon::ThreadPool;
use std::{
    collections::VecDeque,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

/// Block-compressed texture formats supported by the asset pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockFormat {
    /// BC1 linear.
    Bc1,
    /// BC1 sRGB.
    Bc1Srgb,
    /// BC3 linear.
    Bc3,
    /// BC3 sRGB.
    Bc3Srgb,
    /// BC7 linear.
    Bc7,
    /// BC7 sRGB.
    Bc7Srgb,
    /// ASTC 4x4 linear.
    Astc4x4,
    /// ASTC 4x4 sRGB.
    Astc4x4Srgb,
}
impl BlockFormat {
    fn bytes(self) -> usize {
        match self {
            Self::Bc1 | Self::Bc1Srgb => 8,
            _ => 16,
        }
    }
}
/// Nonzero identifier for a streamed texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TextureId(u64);
impl TextureId {
    /// Constructs a nonzero texture identifier.
    ///
    /// # Errors
    ///
    /// Returns [`AssetError::InvalidTextureId`] for zero.
    pub fn try_new(value: u64) -> Result<Self, AssetError> {
        if value == 0 {
            Err(AssetError::InvalidTextureId)
        } else {
            Ok(Self(value))
        }
    }
}
/// Current mip residency state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureState {
    /// No mip is resident.
    Empty,
    /// At least the sample mip is resident.
    SampleReady,
    /// Every mip is resident.
    FullyResident,
}
/// Aligned rectangular texture update region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Region {
    /// Horizontal texel origin.
    pub x: u32,
    /// Vertical texel origin.
    pub y: u32,
    /// Region width in texels.
    pub width: u32,
    /// Region height in texels.
    pub height: u32,
}
impl Region {
    /// Validates a region against texture dimensions and block alignment.
    ///
    /// # Errors
    ///
    /// Returns [`AssetError::Overflow`] for coordinate overflow or
    /// [`AssetError::InvalidRegion`] for zero, out-of-bounds, or misaligned regions.
    pub fn new(
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        tw: u32,
        th: u32,
        _format: BlockFormat,
    ) -> Result<Self, AssetError> {
        if width == 0
            || height == 0
            || x.checked_add(width).ok_or(AssetError::Overflow)? > tw
            || y.checked_add(height).ok_or(AssetError::Overflow)? > th
            || !x.is_multiple_of(4)
            || !y.is_multiple_of(4)
            || (!width.is_multiple_of(4) && x + width != tw)
            || (!height.is_multiple_of(4) && y + height != th)
        {
            return Err(AssetError::InvalidRegion);
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }
}
/// Validates a block-compressed mip payload against its dimensions.
///
/// # Errors
///
/// Returns [`AssetError::InvalidDimensions`] for zero dimensions,
/// [`AssetError::InvalidMip`] for an invalid mip, [`AssetError::Overflow`] for
/// arithmetic overflow, or [`AssetError::InvalidPayload`] when the payload is
/// invalid.
pub fn validate_block_payload(
    format: BlockFormat,
    width: u32,
    height: u32,
    mip: u32,
    payload: &[u8],
) -> Result<(), AssetError> {
    // Zero dimensions do not describe a real mip; rejecting them prevents the
    // `max(1)` normalization below from accepting fabricated 1x1 payloads.
    if width == 0 || height == 0 || mip >= 32 {
        return if width == 0 || height == 0 {
            Err(AssetError::InvalidDimensions)
        } else {
            Err(AssetError::InvalidMip)
        };
    }
    let w = width.checked_shr(mip).ok_or(AssetError::InvalidMip)?.max(1);
    let h = height
        .checked_shr(mip)
        .ok_or(AssetError::InvalidMip)?
        .max(1);
    let bytes_per_block = u32::try_from(format.bytes()).map_err(|_| AssetError::Overflow)?;
    let expected = w
        .div_ceil(4)
        .checked_mul(h.div_ceil(4))
        .and_then(|n| n.checked_mul(bytes_per_block))
        .ok_or(AssetError::Overflow)?;
    let expected = usize::try_from(expected).map_err(|_| AssetError::Overflow)?;
    if payload.len() == expected {
        Ok(())
    } else {
        Err(AssetError::InvalidPayload)
    }
}

/// Mip levels and residency state for one texture.
pub struct MipChain {
    id: TextureId,
    width: u32,
    height: u32,
    format: BlockFormat,
    mips: Vec<Option<Vec<u8>>>,
}
impl MipChain {
    /// Creates an empty mip chain with bounded dimensions and levels.
    ///
    /// # Errors
    ///
    /// Returns [`AssetError::InvalidDimensions`] for zero or unsupported dimensions.
    pub fn new(
        id: TextureId,
        width: u32,
        height: u32,
        mip_count: u32,
        format: BlockFormat,
    ) -> Result<Self, AssetError> {
        if width == 0 || height == 0 || mip_count == 0 || mip_count > 16 {
            return Err(AssetError::InvalidDimensions);
        }
        Ok(Self {
            id,
            width,
            height,
            format,
            mips: vec![None; mip_count as usize],
        })
    }
    /// Returns the number of mip levels.
    pub fn mip_count(&self) -> usize {
        self.mips.len()
    }
    /// Returns the texture identifier.
    pub fn id(&self) -> TextureId {
        self.id
    }
    /// Computes the current residency state.
    pub fn state(&self) -> TextureState {
        let last = self.mips.len() - 1;
        if self.mips[last].is_none() {
            TextureState::Empty
        } else if self.mips.iter().all(Option::is_some) {
            TextureState::FullyResident
        } else {
            TextureState::SampleReady
        }
    }
    /// Uploads one previously missing mip level.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid, already resident, or malformed mip payload.
    pub fn upload_mip(&mut self, mip: u32, payload: Vec<u8>) -> Result<(), AssetError> {
        let index = usize::try_from(mip).map_err(|_| AssetError::InvalidMip)?;
        if index >= self.mips.len() {
            return Err(AssetError::InvalidMip);
        }
        if self.mips[index].is_some() {
            return Err(AssetError::AlreadyResident);
        }
        validate_block_payload(self.format, self.width, self.height, mip, &payload)?;
        self.mips[index] = Some(payload);
        Ok(())
    }
    /// Validates a region update against the resident mip and row pitch.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid mip, region, residency state, row pitch, or payload.
    pub fn update_region(
        &self,
        mip: u32,
        region: Region,
        row_pitch: usize,
        payload: &[u8],
    ) -> Result<(), AssetError> {
        let index = usize::try_from(mip).map_err(|_| AssetError::InvalidMip)?;
        if index >= self.mips.len() {
            return Err(AssetError::InvalidMip);
        }
        let w = (self.width >> mip).max(1);
        let h = (self.height >> mip).max(1);
        Region::new(
            region.x,
            region.y,
            region.width,
            region.height,
            w,
            h,
            self.format,
        )?;
        if self.mips[index].is_none() {
            return Err(AssetError::NotResident);
        }
        let rows = region.height.div_ceil(4) as usize;
        let min_pitch = region.width.div_ceil(4) as usize * self.format.bytes();
        let required = row_pitch.checked_mul(rows).ok_or(AssetError::Overflow)?;
        if row_pitch < min_pitch || payload.len() != required {
            return Err(AssetError::InvalidPayload);
        }
        Ok(())
    }
}

/// Terminal outcome of an asset job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventOutcome {
    /// Job completed successfully.
    Completed,
    /// Job failed.
    Failed,
    /// Job was cancelled.
    Cancelled,
}
/// Pipeline phase that emitted an asset event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventPhase {
    /// Decode phase.
    Decode,
    /// Transcode phase.
    Transcode,
    /// Upload phase.
    Upload,
}
/// Result notification for one asset job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssetEvent {
    /// Correlation identifier assigned by the caller.
    pub correlation_id: u64,
    /// Worker job identifier.
    pub job_id: u64,
    /// Texture associated with the job.
    pub texture: TextureId,
    /// Pipeline phase.
    pub phase: EventPhase,
    /// Number of bytes processed.
    pub bytes: usize,
    /// Terminal job outcome.
    pub outcome: EventOutcome,
    /// Optional failure detail.
    pub error: Option<AssetError>,
}
/// Bounded, cancellable queue of asset events.
pub struct EventQueue {
    events: Mutex<VecDeque<AssetEvent>>,
    capacity: usize,
    reserved: AtomicUsize,
    cancelled: AtomicBool,
}
impl EventQueue {
    /// Creates a queue with the requested event capacity.
    ///
    /// # Errors
    ///
    /// Returns [`AssetError::InvalidQueue`] when capacity is zero.
    pub fn new(capacity: usize) -> Result<Self, AssetError> {
        if capacity == 0 {
            return Err(AssetError::InvalidQueue);
        }
        Ok(Self {
            events: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            reserved: AtomicUsize::new(0),
            cancelled: AtomicBool::new(false),
        })
    }
    /// Enqueues an event unless cancelled or full.
    ///
    /// # Errors
    ///
    /// Returns [`AssetError::Cancelled`] after cancellation or
    /// [`AssetError::QueueFull`] when capacity is exhausted.
    pub fn push(&self, event: AssetEvent) -> Result<(), AssetError> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(AssetError::Cancelled);
        }
        let mut events = self.events.lock();
        if events.len() + self.reserved.load(Ordering::Acquire) >= self.capacity {
            return Err(AssetError::QueueFull);
        }
        events.push_back(event);
        Ok(())
    }
    fn reserve(&self) -> Result<(), AssetError> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(AssetError::Cancelled);
        }
        let events = self.events.lock();
        let reserved = self.reserved.load(Ordering::Acquire);
        if events.len() + reserved >= self.capacity {
            return Err(AssetError::QueueFull);
        }
        self.reserved.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
    fn push_reserved(&self, event: AssetEvent) {
        let mut events = self.events.lock();
        self.reserved.fetch_sub(1, Ordering::AcqRel);
        events.push_back(event);
    }
    /// Removes and returns the oldest queued event.
    pub fn pop(&self) -> Option<AssetEvent> {
        self.events.lock().pop_front()
    }
    /// Prevents future reservations and pushes.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

struct JobPermit {
    jobs: Arc<AtomicUsize>,
    bytes: Arc<AtomicUsize>,
    reserved_bytes: usize,
}
impl Drop for JobPermit {
    fn drop(&mut self) {
        self.jobs.fetch_sub(1, Ordering::AcqRel);
        self.bytes.fetch_sub(self.reserved_bytes, Ordering::AcqRel);
    }
}

/// Bounded CPU worker pool for asset processing.
pub struct CpuPool {
    pool: ThreadPool,
    cancelled: Arc<AtomicBool>,
    max_jobs: usize,
    max_bytes: usize,
    jobs: Arc<AtomicUsize>,
    bytes: Arc<AtomicUsize>,
}
impl CpuPool {
    /// Creates a worker pool with job and byte budgets.
    ///
    /// # Errors
    ///
    /// Returns [`AssetError::InvalidPool`] when any budget is zero or the
    /// worker pool cannot be created.
    pub fn new(threads: usize, max_jobs: usize, max_bytes: usize) -> Result<Self, AssetError> {
        if threads == 0 || max_jobs == 0 || max_bytes == 0 {
            return Err(AssetError::InvalidPool);
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|_| AssetError::InvalidPool)?;
        Ok(Self {
            pool,
            cancelled: Arc::new(AtomicBool::new(false)),
            max_jobs,
            max_bytes,
            jobs: Arc::new(AtomicUsize::new(0)),
            bytes: Arc::new(AtomicUsize::new(0)),
        })
    }
    fn permit(&self, bytes: usize) -> Result<JobPermit, AssetError> {
        if bytes > self.max_bytes {
            return Err(AssetError::QueueFull);
        }
        loop {
            let jobs = self.jobs.load(Ordering::Acquire);
            let used = self.bytes.load(Ordering::Acquire);
            if jobs >= self.max_jobs || used > self.max_bytes - bytes {
                return Err(AssetError::QueueFull);
            }
            if self
                .jobs
                .compare_exchange(jobs, jobs + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                if self
                    .bytes
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| {
                        (v <= self.max_bytes - bytes).then_some(v + bytes)
                    })
                    .is_ok()
                {
                    return Ok(JobPermit {
                        jobs: self.jobs.clone(),
                        bytes: self.bytes.clone(),
                        reserved_bytes: bytes,
                    });
                }
                self.jobs.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }
    /// Schedules a CPU job after reserving its resource budget.
    ///
    /// # Errors
    ///
    /// Returns [`AssetError::Cancelled`] when shut down or
    /// [`AssetError::QueueFull`] when job capacity is exhausted.
    pub fn submit<F>(&self, job: F) -> Result<(), AssetError>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(AssetError::Cancelled);
        }
        let permit = self.permit(0)?;
        let cancelled = self.cancelled.clone();
        self.pool.spawn(move || {
            let _permit = permit;
            if !cancelled.load(Ordering::Acquire) {
                job();
            }
        });
        Ok(())
    }
    /// Runs a job and publishes its result as an asset event.
    ///
    /// # Errors
    ///
    /// Returns queue or worker-budget errors before scheduling the job.
    pub fn submit_event<F>(
        &self,
        queue: Arc<EventQueue>,
        mut event: AssetEvent,
        job: F,
    ) -> Result<(), AssetError>
    where
        F: FnOnce() -> Result<usize, AssetError> + Send + 'static,
    {
        queue.reserve()?;
        let permit = match self.permit(event.bytes) {
            Ok(p) => p,
            Err(e) => {
                queue.reserved.fetch_sub(1, Ordering::AcqRel);
                return Err(e);
            }
        };
        let cancelled = self.cancelled.clone();
        self.pool.spawn(move || {
            let _permit = permit;
            let result = if cancelled.load(Ordering::Acquire) {
                Err(AssetError::Cancelled)
            } else {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(job))
                    .map_err(|_| AssetError::WorkerPanic)
                    .and_then(|r| r)
            };
            match result {
                Ok(bytes) => {
                    event.bytes = bytes;
                    event.outcome = EventOutcome::Completed;
                    event.error = None;
                }
                Err(error) => {
                    event.outcome = if error == AssetError::Cancelled {
                        EventOutcome::Cancelled
                    } else {
                        EventOutcome::Failed
                    };
                    event.error = Some(error);
                }
            }
            queue.push_reserved(event);
        });
        Ok(())
    }
    /// Cancels queued and future CPU work.
    pub fn shutdown(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

/// Transcodes a validated Basis payload into the requested block format.
///
/// # Errors
///
/// Returns [`AssetError::InvalidBasis`] for invalid headers or
/// [`AssetError::BasisTranscode`] when transcoding fails.
#[cfg(feature = "basis")]
pub fn transcode_basis(data: &[u8], target: BlockFormat) -> Result<Vec<u8>, AssetError> {
    use basis_universal::transcoding::{Transcoder, TranscoderTextureFormat};
    let mut transcoder = Transcoder::new();
    if !transcoder.validate_header(data) {
        return Err(AssetError::InvalidBasis);
    }
    let format = match target {
        BlockFormat::Bc1 | BlockFormat::Bc1Srgb => TranscoderTextureFormat::BC1_RGB,
        BlockFormat::Bc3 | BlockFormat::Bc3Srgb => TranscoderTextureFormat::BC3_RGBA,
        BlockFormat::Bc7 | BlockFormat::Bc7Srgb => TranscoderTextureFormat::BC7_RGBA,
        BlockFormat::Astc4x4 | BlockFormat::Astc4x4Srgb => TranscoderTextureFormat::ASTC_4x4_RGBA,
    };
    transcoder
        .prepare_transcoding(data)
        .map_err(|()| AssetError::BasisTranscode)?;
    transcoder
        .transcode_image_level(
            data,
            format,
            basis_universal::transcoding::TranscodeParameters::default(),
        )
        .map_err(|_| AssetError::BasisTranscode)
}

#[cfg(not(feature = "basis"))]
/// Transcodes Basis payload bytes into the requested block format.
///
/// # Errors
///
/// Returns [`AssetError::BasisDisabled`] when Basis support is unavailable.
pub fn transcode_basis(_data: &[u8], _target: BlockFormat) -> Result<Vec<u8>, AssetError> {
    Err(AssetError::BasisDisabled)
}

#[derive(Debug, Eq, PartialEq)]
/// Parsed KTX2 payload representation.
pub enum Ktx2Payload {
    /// Direct block-compressed mip levels.
    Direct {
        /// Block format of each level.
        format: BlockFormat,
        /// Texture width.
        width: u32,
        /// Texture height.
        height: u32,
        /// Mip payloads in level order.
        levels: Vec<Vec<u8>>,
    },
    /// Basis-compressed payload requiring runtime transcoding.
    Basis {
        /// Texture width.
        width: u32,
        /// Texture height.
        height: u32,
        /// Basis payload bytes.
        data: Vec<u8>,
    },
}
/// Parses and validates a KTX2 container.
///
/// # Errors
///
/// Returns a parsing, bounds, format, or payload validation error.
///
/// # Panics
///
/// Panics only if a slice accepted as exactly four or eight bytes cannot be converted to its fixed-size array type.
pub fn parse_ktx2(input: &[u8]) -> Result<Ktx2Payload, AssetError> {
    const IDENT: &[u8; 12] = b"\xabKTX 20\xbb\r\n\x1a\n";
    if input.len() < 80 || &input[..12] != IDENT {
        return Err(AssetError::InvalidKtx2);
    }
    let read32 = |at: usize| {
        input
            .get(at..at + 4)
            .ok_or(AssetError::Truncated)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    let read64 = |at: usize| {
        input
            .get(at..at + 8)
            .ok_or(AssetError::Truncated)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    };
    let format = read32(12)?;
    let width = read32(20)?;
    let height = read32(24)?;
    let layers = read32(32)?;
    let faces = read32(36)?;
    let levels = read32(40)?;
    let supercompression = read32(44)?;
    if width == 0 || height == 0 || levels == 0 || levels > 16 || layers > 1 || faces != 1 {
        return Err(AssetError::InvalidKtx2);
    }
    if format == 0 {
        if supercompression != 1 {
            return Err(AssetError::UnsupportedKtx2);
        }
        return Ok(Ktx2Payload::Basis {
            width,
            height,
            data: input.to_vec(),
        });
    }
    if supercompression != 0 {
        return Err(AssetError::UnsupportedKtx2);
    }
    let format = match format {
        131 => BlockFormat::Bc1,
        132 => BlockFormat::Bc1Srgb,
        137 => BlockFormat::Bc3,
        138 => BlockFormat::Bc3Srgb,
        145 => BlockFormat::Bc7,
        146 => BlockFormat::Bc7Srgb,
        157 => BlockFormat::Astc4x4,
        158 => BlockFormat::Astc4x4Srgb,
        _ => return Err(AssetError::UnsupportedKtx2),
    };
    let table = 80usize
        .checked_add(
            (levels as usize)
                .checked_mul(24)
                .ok_or(AssetError::Overflow)?,
        )
        .ok_or(AssetError::Overflow)?;
    if table > input.len() {
        return Err(AssetError::Truncated);
    }
    let mut output = Vec::new();
    for level in 0..levels as usize {
        let at = 80 + level * 24;
        let offset = usize::try_from(read64(at)?).map_err(|_| AssetError::Overflow)?;
        let length = usize::try_from(read64(at + 8)?).map_err(|_| AssetError::Overflow)?;
        let end = offset.checked_add(length).ok_or(AssetError::Overflow)?;
        if offset < table || end > input.len() {
            return Err(AssetError::Truncated);
        }
        validate_block_payload(
            format,
            width,
            height,
            u32::try_from(level).map_err(|_| AssetError::Overflow)?,
            &input[offset..end],
        )?;
        output.push(input[offset..end].to_vec());
    }
    Ok(Ktx2Payload::Direct {
        format,
        width,
        height,
        levels: output,
    })
}

/// Errors reported by bounded asset parsing and processing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssetError {
    /// Input ended before a complete record.
    Truncated,
    /// Texture ID is zero.
    InvalidTextureId,
    /// Texture dimensions or mip count are invalid.
    InvalidDimensions,
    /// Mip index is invalid.
    InvalidMip,
    /// Region coordinates or alignment are invalid.
    InvalidRegion,
    /// Mip payload length is invalid.
    InvalidPayload,
    /// Arithmetic exceeded a representable bound.
    Overflow,
    /// Mip is already resident.
    AlreadyResident,
    /// Mip is not resident.
    NotResident,
    /// Queue has been cancelled.
    Cancelled,
    /// Queue has no free capacity.
    QueueFull,
    /// Worker pool has shut down.
    Shutdown,
    /// Queue capacity is invalid.
    InvalidQueue,
    /// Worker pool limits are invalid.
    InvalidPool,
    /// Worker closure panicked.
    WorkerPanic,
    /// Basis support was not compiled.
    BasisDisabled,
    /// Basis transcoding failed.
    BasisTranscode,
    /// Basis payload is invalid.
    InvalidBasis,
    /// KTX2 payload is invalid.
    InvalidKtx2,
    /// KTX2 format is unsupported.
    UnsupportedKtx2,
}
impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for AssetError {}
