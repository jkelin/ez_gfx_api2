use ez_gfx_core::{Backend, capability::SemanticProfile};

pub const MAX_CACHE_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
const MAGIC: &[u8; 8] = b"EZPCACHE";
const VERSION: u16 = 1;
const FIXED_HEADER: usize = 74;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PipelineKey {
    shader_digest: [u8; 32],
    interface_digest: [u8; 32],
    backend: Backend,
    state_schema: u32,
    profile_schema: u32,
}

impl PipelineKey {
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
pub struct CacheIdentity {
    backend: Backend,
    adapter: [u8; 16],
    driver: String,
    profile_schema: u32,
    cache_schema: u32,
}

impl CacheIdentity {
    /// Zero device identity, empty/oversized driver identity, and schema zero are invalidation hazards.
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
pub struct PipelineCacheEnvelope {
    identity: CacheIdentity,
    payload: Vec<u8>,
}

impl PipelineCacheEnvelope {
    /// Empty native blobs are valid, but payloads above the fixed host/runtime boundary are rejected.
    pub fn new(identity: CacheIdentity, payload: Vec<u8>) -> Result<Self, CacheError> {
        if payload.len() > MAX_CACHE_PAYLOAD_BYTES {
            return Err(CacheError::TooLarge);
        }
        Ok(Self { identity, payload })
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
    pub fn identity(&self) -> &CacheIdentity {
        &self.identity
    }

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
        output.extend_from_slice(&(self.identity.driver.len() as u16).to_le_bytes());
        output.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        output.extend_from_slice(&checksum);
        output.extend_from_slice(self.identity.driver.as_bytes());
        output.extend_from_slice(&self.payload);
        Ok(output)
    }

    /// Parsing is bounded and validates complete identity before exposing opaque native bytes.
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

fn checksum(identity: &CacheIdentity, payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ez-gfx-pipeline-cache-v1");
    hasher.update(&[backend_byte(identity.backend)]);
    hasher.update(&identity.adapter);
    hasher.update(&identity.profile_schema.to_le_bytes());
    hasher.update(&identity.cache_schema.to_le_bytes());
    hasher.update(&(identity.driver.len() as u16).to_le_bytes());
    hasher.update(identity.driver.as_bytes());
    hasher.update(&(payload.len() as u32).to_le_bytes());
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

fn backend_byte(backend: Backend) -> u8 {
    backend as u8
}
fn parse_backend(value: u8) -> Result<Backend, CacheError> {
    match value {
        1 => Ok(Backend::Vulkan),
        2 => Ok(Backend::Dx12),
        3 => Ok(Backend::Metal),
        _ => Err(CacheError::InvalidHeader),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheError {
    InvalidIdentity,
    TooLarge,
    Truncated,
    InvalidHeader,
    IdentityMismatch,
    ChecksumMismatch,
}
