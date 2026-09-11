const ALLOCATION_BLOCK_ALIGNMENT: u64 = 4 * 1024 * 1024;
/// Portable byte offset of the element array in a counter buffer.
///
/// The count occupies the first four bytes; the gap keeps the element descriptor
/// aligned for every supported storage-buffer backend.
pub const COUNTER_BUFFER_ELEMENT_OFFSET: u64 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Initial and maximum allocator block sizes for device-local and host-visible memory.
pub struct AllocationBlockPolicy {
    /// Initial device-local block size in bytes.
    pub initial_device: u64,
    /// Maximum device-local block size in bytes.
    pub maximum_device: u64,
    /// Initial host-visible block size in bytes.
    pub initial_host: u64,
    /// Maximum host-visible block size in bytes.
    pub maximum_host: u64,
}

impl AllocationBlockPolicy {
    /// Creates a block policy after validating sizes and alignment.
    ///
    /// # Errors
    ///
    /// Returns an error when a size is zero, not 4 MiB aligned, or exceeds its maximum.
    pub const fn new(
        initial_device: u64,
        maximum_device: u64,
        initial_host: u64,
        maximum_host: u64,
    ) -> Result<Self, AllocationBlockPolicyError> {
        // gpu-allocator accepts only 4 MiB block increments; reject instead of allowing it to clamp.
        if initial_device == 0 || maximum_device == 0 || initial_host == 0 || maximum_host == 0 {
            return Err(AllocationBlockPolicyError::ZeroSize);
        }
        if !initial_device.is_multiple_of(ALLOCATION_BLOCK_ALIGNMENT)
            || !maximum_device.is_multiple_of(ALLOCATION_BLOCK_ALIGNMENT)
            || !initial_host.is_multiple_of(ALLOCATION_BLOCK_ALIGNMENT)
            || !maximum_host.is_multiple_of(ALLOCATION_BLOCK_ALIGNMENT)
        {
            return Err(AllocationBlockPolicyError::InvalidAlignment);
        }
        if initial_device > maximum_device || initial_host > maximum_host {
            return Err(AllocationBlockPolicyError::InvalidRange);
        }
        Ok(Self {
            initial_device,
            maximum_device,
            initial_host,
            maximum_host,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Reasons an allocation block policy is invalid.
pub enum AllocationBlockPolicyError {
    /// A configured block size is zero.
    ZeroSize,
    /// A configured block size is not 4 MiB aligned.
    InvalidAlignment,
    /// An initial size exceeds its maximum.
    InvalidRange,
}

/// Default allocation block sizing policy.
pub const DEFAULT_ALLOCATION_BLOCK_POLICY: AllocationBlockPolicy = AllocationBlockPolicy {
    initial_device: 16 * 1024 * 1024,
    maximum_device: 256 * 1024 * 1024,
    initial_host: 8 * 1024 * 1024,
    maximum_host: 64 * 1024 * 1024,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// On-demand allocator telemetry mapped from `generate_report`.
///
/// All sizes accumulate with saturation and never wrap. Query it only through
/// explicit diagnostics paths: report generation itself allocates and must
/// never run per frame.
pub struct AllocatorTelemetry {
    /// Bytes sub-allocated to live resources.
    pub live_bytes: u64,
    /// Bytes committed in allocator blocks, including unallocated regions.
    pub block_bytes: u64,
    /// Committed allocator blocks.
    pub block_count: u32,
    /// Live sub-allocations.
    pub allocation_count: u32,
}

impl AllocatorTelemetry {
    /// Committed but unallocated bytes: block capacity minus live bytes.
    pub const fn waste_bytes(&self) -> u64 {
        self.block_bytes.saturating_sub(self.live_bytes)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// On-demand backend memory telemetry; zero means unknown at context level.
///
/// Swapchain and depth sizes are resolution-scaled RGBA8/D32 estimates, not
/// driver measurements: four bytes per pixel times image count or one depth
/// target. Query explicitly; never per frame.
pub struct BackendMemoryTelemetry {
    /// Allocator telemetry, or `None` when the backend has no allocator yet.
    pub allocator: Option<AllocatorTelemetry>,
    /// Swapchain color images retained by the backend.
    pub swapchain_images: u32,
    /// Current swapchain extent as `(width, height)`.
    pub swapchain_extent: (u32, u32),
    /// Raw backend swapchain format code; zero means unknown at context level.
    pub swapchain_format: u32,
    /// Estimated swapchain storage bytes (images times extent times four).
    pub swapchain_bytes: u64,
    /// Estimated depth-target bytes (extent times four when present).
    pub depth_bytes: u64,
    /// Frame slots retained by the backend.
    pub frame_slots: u32,
}

/// Estimates RGBA8 storage for `images` pictures of `extent`, saturating.
///
/// Backends use this for swapchain byte estimates so every estimate shares one
/// saturating convention instead of ad-hoc checked arithmetic.
pub const fn rgba8_image_bytes(images: u32, width: u32, height: u32) -> u64 {
    // Each factor widens before multiplying so no intermediate `u32` product wraps.
    (images as u64)
        .saturating_mul(width as u64)
        .saturating_mul(height as u64)
        .saturating_mul(4)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Shared limits for reusable staging and adaptive transfer batches.
pub struct StagingPolicy {
    /// Smallest power-of-two staging bucket.
    pub minimum_bucket_bytes: u64,
    /// Largest accepted staging allocation.
    pub maximum_bucket_bytes: u64,
    /// Byte threshold that flushes a transfer batch.
    pub batch_bytes: u64,
    /// Copy-count threshold that flushes a transfer batch.
    pub batch_copies: usize,
}

impl StagingPolicy {
    /// Creates a staging policy with power-of-two buckets and nonzero batch limits.
    ///
    /// # Errors
    ///
    /// Returns [`StagingPolicyError::InvalidPolicy`] for zero, non-power-of-two, or inverted limits.
    pub const fn new(
        minimum_bucket_bytes: u64,
        maximum_bucket_bytes: u64,
        batch_bytes: u64,
        batch_copies: usize,
    ) -> Result<Self, StagingPolicyError> {
        if minimum_bucket_bytes == 0
            || maximum_bucket_bytes == 0
            || batch_bytes == 0
            || batch_copies == 0
            || !minimum_bucket_bytes.is_power_of_two()
            || !maximum_bucket_bytes.is_power_of_two()
            || minimum_bucket_bytes > maximum_bucket_bytes
        {
            return Err(StagingPolicyError::InvalidPolicy);
        }
        Ok(Self {
            minimum_bucket_bytes,
            maximum_bucket_bytes,
            batch_bytes,
            batch_copies,
        })
    }

    /// Reports whether an accumulated batch reached either configured limit.
    pub const fn should_flush(self, copies: usize, bytes: u64) -> bool {
        copies >= self.batch_copies || bytes >= self.batch_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Invalid staging-policy or staging-size request.
pub enum StagingPolicyError {
    /// Policy limits are zero, inverted, or not powers of two.
    InvalidPolicy,
    /// A staging request is zero or exceeds the configured maximum.
    InvalidSize,
}

/// Returns the smallest configured power-of-two bucket containing `bytes`.
///
/// # Errors
///
/// Returns [`StagingPolicyError::InvalidSize`] when `bytes` is zero, exceeds the maximum, or cannot round up.
pub const fn staging_bucket_size(
    bytes: u64,
    policy: StagingPolicy,
) -> Result<u64, StagingPolicyError> {
    if bytes == 0 || bytes > policy.maximum_bucket_bytes {
        return Err(StagingPolicyError::InvalidSize);
    }
    let requested = if bytes < policy.minimum_bucket_bytes {
        policy.minimum_bucket_bytes
    } else {
        bytes
    };
    match requested.checked_next_power_of_two() {
        Some(bucket) if bucket <= policy.maximum_bucket_bytes => Ok(bucket),
        _ => Err(StagingPolicyError::InvalidSize),
    }
}

/// Default staging and transfer batching limits.
pub const DEFAULT_STAGING_POLICY: StagingPolicy = StagingPolicy {
    minimum_bucket_bytes: 64 * 1024,
    maximum_bucket_bytes: 64 * 1024 * 1024,
    batch_bytes: 32 * 1024 * 1024,
    batch_copies: 64,
};

/// Retention ceiling for the shared upload staging pool.
///
/// Idle trimming already evicts stale buckets, but without a budget one
/// pathological frame pins its peak forever; pressure trims enforce this only
/// through explicit calls, never implicitly inside `put`.
pub const DEFAULT_SHARED_STAGING_BUDGET: u64 = 32 * 1024 * 1024;

/// Retention ceiling for each per-stride buffer staging pool.
///
/// The buffer pool map holds one pool per element stride, so this bounds each
/// entry rather than their sum; the context aggregate high-water observes the sum.
pub const DEFAULT_BUFFER_STAGING_BUDGET: u64 = 16 * 1024 * 1024;

/// Retention ceiling for the counter staging pool.
///
/// Counter payloads are kilobytes at steady state; this ceiling only bites after
/// pathological multi-megabyte writes that idle trimming has not yet swept.
pub const DEFAULT_COUNTER_STAGING_BUDGET: u64 = 8 * 1024 * 1024;

/// Context-wide ceiling for summed staging retention across every pool.
///
/// Per-pool ceilings bound local growth, but the buffer map holds one pool per
/// stride, so only a global cap bounds the aggregate: enforcement evicts the
/// largest completed bucket across all pools until the summed retention fits.
/// In-flight buckets are never candidates and may keep the total over budget
/// until the next enforcement call revisits them.
pub const DEFAULT_STAGING_AGGREGATE_BUDGET: u64 = 64 * 1024 * 1024;
