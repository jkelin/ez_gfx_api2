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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockFormat {
    Bc1,
    Bc1Srgb,
    Bc3,
    Bc3Srgb,
    Bc7,
    Bc7Srgb,
    Astc4x4,
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TextureId(u64);
impl TextureId {
    pub fn try_new(value: u64) -> Result<Self, AssetError> {
        if value == 0 {
            Err(AssetError::InvalidTextureId)
        } else {
            Ok(Self(value))
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureState {
    Empty,
    SampleReady,
    FullyResident,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
impl Region {
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
            || x % 4 != 0
            || y % 4 != 0
            || (width % 4 != 0 && x + width != tw)
            || (height % 4 != 0 && y + height != th)
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
pub fn validate_block_payload(
    format: BlockFormat,
    width: u32,
    height: u32,
    mip: u32,
    payload: &[u8],
) -> Result<(), AssetError> {
    if mip >= 32 {
        return Err(AssetError::InvalidMip);
    };
    let w = width.checked_shr(mip).ok_or(AssetError::InvalidMip)?.max(1);
    let h = height
        .checked_shr(mip)
        .ok_or(AssetError::InvalidMip)?
        .max(1);
    let expected = w
        .div_ceil(4)
        .checked_mul(h.div_ceil(4))
        .and_then(|n| n.checked_mul(format.bytes() as u32))
        .ok_or(AssetError::Overflow)? as usize;
    if payload.len() != expected {
        Err(AssetError::InvalidPayload)
    } else {
        Ok(())
    }
}

pub struct MipChain {
    id: TextureId,
    width: u32,
    height: u32,
    format: BlockFormat,
    mips: Vec<Option<Vec<u8>>>,
}
impl MipChain {
    pub fn new(
        id: TextureId,
        width: u32,
        height: u32,
        mip_count: u32,
        format: BlockFormat,
    ) -> Result<Self, AssetError> {
        if width == 0 || height == 0 || mip_count == 0 || mip_count > 16 {
            return Err(AssetError::InvalidDimensions);
        };
        Ok(Self {
            id,
            width,
            height,
            format,
            mips: vec![None; mip_count as usize],
        })
    }
    pub fn mip_count(&self) -> usize {
        self.mips.len()
    }
    pub fn id(&self) -> TextureId {
        self.id
    }
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
    pub fn upload_mip(&mut self, mip: u32, payload: Vec<u8>) -> Result<(), AssetError> {
        let index = usize::try_from(mip).map_err(|_| AssetError::InvalidMip)?;
        if index >= self.mips.len() {
            return Err(AssetError::InvalidMip);
        };
        if self.mips[index].is_some() {
            return Err(AssetError::AlreadyResident);
        };
        validate_block_payload(self.format, self.width, self.height, mip, &payload)?;
        self.mips[index] = Some(payload);
        Ok(())
    }
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
        };
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
        };
        let rows = region.height.div_ceil(4) as usize;
        let min_pitch = region.width.div_ceil(4) as usize * self.format.bytes();
        let required = row_pitch.checked_mul(rows).ok_or(AssetError::Overflow)?;
        if row_pitch < min_pitch || payload.len() != required {
            return Err(AssetError::InvalidPayload);
        };
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventOutcome {
    Completed,
    Failed,
    Cancelled,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventPhase {
    Decode,
    Transcode,
    Upload,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssetEvent {
    pub correlation_id: u64,
    pub job_id: u64,
    pub texture: TextureId,
    pub phase: EventPhase,
    pub bytes: usize,
    pub outcome: EventOutcome,
    pub error: Option<AssetError>,
}
pub struct EventQueue {
    events: Mutex<VecDeque<AssetEvent>>,
    capacity: usize,
    reserved: AtomicUsize,
    cancelled: AtomicBool,
}
impl EventQueue {
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
    pub fn pop(&self) -> Option<AssetEvent> {
        self.events.lock().pop_front()
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release)
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

pub struct CpuPool {
    pool: ThreadPool,
    cancelled: Arc<AtomicBool>,
    max_jobs: usize,
    max_bytes: usize,
    jobs: Arc<AtomicUsize>,
    bytes: Arc<AtomicUsize>,
}
impl CpuPool {
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
    pub fn shutdown(&self) {
        self.cancelled.store(true, Ordering::Release)
    }
}

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
        .map_err(|_| AssetError::BasisTranscode)?;
    transcoder
        .transcode_image_level(data, format, Default::default())
        .map_err(|_| AssetError::BasisTranscode)
}

#[cfg(not(feature = "basis"))]
pub fn transcode_basis(_data: &[u8], _target: BlockFormat) -> Result<Vec<u8>, AssetError> {
    Err(AssetError::BasisDisabled)
}

#[derive(Debug, Eq, PartialEq)]
pub enum Ktx2Payload {
    Direct {
        format: BlockFormat,
        width: u32,
        height: u32,
        levels: Vec<Vec<u8>>,
    },
    Basis {
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
}
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
        };
        validate_block_payload(format, width, height, level as u32, &input[offset..end])?;
        output.push(input[offset..end].to_vec());
    }
    Ok(Ktx2Payload::Direct {
        format,
        width,
        height,
        levels: output,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssetError {
    Truncated,
    InvalidTextureId,
    InvalidDimensions,
    InvalidMip,
    InvalidRegion,
    InvalidPayload,
    Overflow,
    AlreadyResident,
    NotResident,
    Cancelled,
    QueueFull,
    Shutdown,
    InvalidQueue,
    InvalidPool,
    WorkerPanic,
    BasisDisabled,
    BasisTranscode,
    InvalidBasis,
    InvalidKtx2,
    UnsupportedKtx2,
}
impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for AssetError {}
