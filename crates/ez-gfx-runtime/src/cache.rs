use ez_gfx_core::{Backend, capability::SemanticProfile};

/// Maximum encoded native cache payload size accepted by the runtime.
pub const MAX_CACHE_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
const MAGIC: &[u8; 8] = b"EZPCACHE";
const VERSION: u16 = 1;
const FIXED_HEADER: usize = 74;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
/// Stable inputs that distinguish a compiled graphics pipeline.
pub struct PipelineKey {
    /// Digest of the pipeline's shader code.
    shader_digest: [u8; 32],
    /// Digest of the shader-resource interface.
    interface_digest: [u8; 32],
    /// Graphics API targeted by the pipeline.
    backend: Backend,
    /// Schema version for serialized pipeline state.
    state_schema: u32,
    /// Schema version for semantic profile data.
    profile_schema: u32,
}

impl PipelineKey {
    /// Creates a key from pipeline digests, backend, and schema versions.
    pub const fn new(
        shader_digest: [u8; 32],
        interface_digest: [u8; 32],
        backend: Backend,
        state_schema: u32,
        profile_schema: u32,
    ) -> Self {
        Self {
            shader_digest,
            interface_digest,
            backend,
            state_schema,
            profile_schema,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Device and schema metadata required to reuse a native pipeline cache.
pub struct CacheIdentity {
    /// Graphics API that produced the cache.
    backend: Backend,
    /// Stable 16-byte adapter identifier.
    adapter: [u8; 16],
    /// Driver identity associated with the cached data.
    driver: String,
    /// Semantic profile schema version used to create the cache.
    profile_schema: u32,
    /// Application cache schema version used to create the cache.
    cache_schema: u32,
}

impl CacheIdentity {
    /// Zero device identity, empty/oversized driver identity, and schema zero are invalidation hazards.
    ///
    /// # Errors
    ///
    /// Returns `CacheError::InvalidIdentity` if the adapter is zero, the driver is empty, longer than 255 bytes, or contains a NUL byte, or the cache schema is zero.
    pub fn new(
        backend: Backend,
        adapter: [u8; 16],
        driver: impl Into<String>,
        profile: SemanticProfile,
        cache_schema: u32,
    ) -> Result<Self, CacheError> {
        let driver = driver.into();
        if adapter == [0; 16]
            || driver.is_empty()
            || driver.len() > 255
            || driver.as_bytes().contains(&0)
            || cache_schema == 0
        {
            return Err(CacheError::InvalidIdentity);
        }
        Ok(Self {
            backend,
            adapter,
            driver,
            profile_schema: profile.schema_version(),
            cache_schema,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Native pipeline cache bytes paired with their device and schema identity.
pub struct PipelineCacheEnvelope {
    /// Metadata governing compatibility of the cached bytes.
    identity: CacheIdentity,
    /// Backend-native pipeline cache bytes.
    payload: Vec<u8>,
}

impl PipelineCacheEnvelope {
    /// Empty native blobs are valid, but payloads above the fixed host/runtime boundary are rejected.
    ///
    /// # Errors
    ///
    /// Returns `CacheError::TooLarge` if the payload exceeds `MAX_CACHE_PAYLOAD_BYTES`.
    pub fn new(identity: CacheIdentity, payload: Vec<u8>) -> Result<Self, CacheError> {
        if payload.len() > MAX_CACHE_PAYLOAD_BYTES {
            return Err(CacheError::TooLarge);
        }
        Ok(Self { identity, payload })
    }

    /// Returns the backend-native pipeline cache bytes.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
    /// Returns the metadata governing cache compatibility.
    pub fn identity(&self) -> &CacheIdentity {
        &self.identity
    }

    /// Encodes the identity and native cache bytes into the versioned binary format.
    ///
    /// # Errors
    ///
    /// Returns `CacheError::TooLarge` if the encoded size overflows, the payload exceeds supported bounds, or the driver or payload length cannot be encoded.
    pub fn encode(&self) -> Result<Vec<u8>, CacheError> {
        let total = FIXED_HEADER
            .checked_add(self.identity.driver.len())
            .and_then(|value| value.checked_add(self.payload.len()))
            .ok_or(CacheError::TooLarge)?;
        if self.payload.len() > MAX_CACHE_PAYLOAD_BYTES {
            return Err(CacheError::TooLarge);
        }
        let checksum = checksum(&self.identity, &self.payload);
        let mut output = Vec::with_capacity(total);
        output.extend_from_slice(MAGIC);
        output.extend_from_slice(&VERSION.to_le_bytes());
        output.push(backend_byte(self.identity.backend));
        output.push(0);
        output.extend_from_slice(&self.identity.cache_schema.to_le_bytes());
        output.extend_from_slice(&self.identity.profile_schema.to_le_bytes());
        output.extend_from_slice(&self.identity.adapter);
        let driver_len =
            u16::try_from(self.identity.driver.len()).map_err(|_| CacheError::TooLarge)?;
        let payload_len = u32::try_from(self.payload.len()).map_err(|_| CacheError::TooLarge)?;
        output.extend_from_slice(&driver_len.to_le_bytes());
        output.extend_from_slice(&payload_len.to_le_bytes());
        output.extend_from_slice(&checksum);
        output.extend_from_slice(self.identity.driver.as_bytes());
        output.extend_from_slice(&self.payload);
        Ok(output)
    }
    /// # Errors
    ///
    /// Returns an error when the cache header, identity, payload, or checksum is invalid.
    ///
    /// # Panics
    ///
    /// Panics only if the fixed-size header slices violate the length checks above.
    pub fn decode(bytes: &[u8], expected: &CacheIdentity) -> Result<Self, CacheError> {
        if bytes.len() < FIXED_HEADER {
            return Err(CacheError::Truncated);
        }
        if &bytes[..8] != MAGIC || u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != VERSION {
            return Err(CacheError::InvalidHeader);
        }
        let backend = parse_backend(bytes[10])?;
        if bytes[11] != 0 {
            return Err(CacheError::InvalidHeader);
        }
        let cache_schema = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let profile_schema = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let mut adapter = [0; 16];
        adapter.copy_from_slice(&bytes[20..36]);
        let driver_len = u16::from_le_bytes(bytes[36..38].try_into().unwrap()) as usize;
        let payload_len = u32::from_le_bytes(bytes[38..42].try_into().unwrap()) as usize;
        if payload_len > MAX_CACHE_PAYLOAD_BYTES {
            return Err(CacheError::TooLarge);
        }
        let end = FIXED_HEADER
            .checked_add(driver_len)
            .and_then(|value| value.checked_add(payload_len))
            .ok_or(CacheError::TooLarge)?;
        if end != bytes.len() {
            return Err(if end > bytes.len() {
                CacheError::Truncated
            } else {
                CacheError::InvalidHeader
            });
        }
        let driver = std::str::from_utf8(&bytes[FIXED_HEADER..FIXED_HEADER + driver_len])
            .map_err(|_| CacheError::InvalidIdentity)?
            .to_owned();
        let identity = CacheIdentity {
            backend,
            adapter,
            driver,
            profile_schema,
            cache_schema,
        };
        if &identity != expected {
            return Err(CacheError::IdentityMismatch);
        }
        let payload = bytes[FIXED_HEADER + driver_len..].to_vec();
        let mut stored = [0; 32];
        stored.copy_from_slice(&bytes[42..74]);
        if stored != checksum(&identity, &payload) {
            return Err(CacheError::ChecksumMismatch);
        }
        Ok(Self { identity, payload })
    }
}

/// Computes the BLAKE3 digest covering cache identity and payload bytes.
fn checksum(identity: &CacheIdentity, payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ez-gfx-pipeline-cache-v1");
    hasher.update(&[backend_byte(identity.backend)]);
    hasher.update(&identity.adapter);
    hasher.update(&identity.profile_schema.to_le_bytes());
    hasher.update(&identity.cache_schema.to_le_bytes());
    let driver_len =
        u16::try_from(identity.driver.len()).expect("cache identity driver length is bounded");
    hasher.update(&driver_len.to_le_bytes());
    hasher.update(identity.driver.as_bytes());
    let payload_len = u32::try_from(payload.len()).expect("cache payload length is bounded");
    hasher.update(&payload_len.to_le_bytes());
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

/// Encodes a graphics backend as its cache-header byte.
fn backend_byte(backend: Backend) -> u8 {
    backend as u8
}
/// Decodes a cache-header byte into a supported graphics backend.
///
/// # Errors
///
/// Returns `CacheError::InvalidHeader` if the byte does not identify a supported graphics backend.
fn parse_backend(value: u8) -> Result<Backend, CacheError> {
    match value {
        1 => Ok(Backend::Vulkan),
        2 => Ok(Backend::Dx12),
        3 => Ok(Backend::Metal),
        _ => Err(CacheError::InvalidHeader),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Reasons pipeline cache creation, encoding, or decoding can fail.
pub enum CacheError {
    /// Device, driver, or schema metadata is not valid for cache reuse.
    InvalidIdentity,
    /// The payload or encoded size exceeds supported bounds.
    TooLarge,
    /// The encoded cache ends before its declared content is complete.
    Truncated,
    /// The cache magic, version, reserved byte, backend, or length is invalid.
    InvalidHeader,
    /// Encoded device or schema metadata differs from the expected identity.
    IdentityMismatch,
    /// The stored digest does not match the encoded identity and payload.
    ChecksumMismatch,
}
